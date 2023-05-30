import pytorch_lightning as pl
import torch.nn as nn
from layers.Mol2Graph import Mol2Graph
from utils.loss_fns import loss_fn_map
from utils.activation_fns import activation_fn_map
from utils.EMA import ExponentialMovingAverage
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
from layers.threeEDis.threeEDis_output import ThreeOrderOutputBlock
from layers.threeEDis.threeEDis_init import ThreeEDisInit
from layers.threeEDis.threeEDis_interaction import ThreeEDisLayer
from layers.twoFDis.twoFDis_init import TwoFDisInit
from layers.twoFDis.twoFDis_interaction import TwoFDisLayer
from layers.twoFDis.twoFDis_output import TwoOrderOutputBlock, TwoOrderDipOutputBlock, TwoOrderElcOutputBlock
from layers.twoEDis.twoEDis_init import TwoEDisInit
from layers.twoEDis.twoEDis_interaction import TwoEDisLayer

from layers.basic_layers import Residual, Dense
from utils.GradualWarmupScheduler import GradualWarmupScheduler

init_layer_dict = {
    "2FDis": TwoFDisInit,
    "3EDis": ThreeEDisInit,
    "2EDis": TwoEDisInit,
}

interaction_layer_dict = {
    "2FDis": TwoFDisLayer,
    "3EDis": ThreeEDisLayer,
    "2EDis": TwoEDisLayer,
}

output_layer_dict = {
    "2FDis": TwoOrderOutputBlock,
    "2FDisDip": TwoOrderDipOutputBlock,
    "2FDisElc": TwoOrderElcOutputBlock,
    "3EDis": ThreeOrderOutputBlock,
    "2EDis": TwoOrderOutputBlock,
    "3FDis": ThreeOrderOutputBlock
}


class LitkDis(pl.LightningModule):
    def __init__(
        self, 
        model_name: str,
        model_config: dict,
        optimizer_config: dict,
        scheduler_config: dict,
        qm9: bool = False,
        global_y_std: float = 1.,
        global_y_mean: float = 0.,
        **kwargs
        ):
        
        super().__init__()
        self.save_hyperparameters()
        
        init_layer = init_layer_dict[model_name]
        interaction_layer = interaction_layer_dict[model_name]
        
        if qm9 and model_name == "2FDis" and (int(kwargs["data_name"]) == 0):
            output_layer = output_layer_dict["2FDisDip"]
            global_y_std = 1
            global_y_mean = 0
            print("Using 2FDisDip as output layer")
        elif qm9 and model_name == "2FDis" and (int(kwargs["data_name"]) == 5):
            output_layer = output_layer_dict["2FDisElc"]
            global_y_std = 1
            global_y_mean = 0
            print("Using 2FDisElc as output layer")
        else:
            output_layer = output_layer_dict[model_name]
            
        
        '''
            model configs
        '''

        rbf = model_config.rbf
        max_z = model_config.max_z
        rbf_trainable = model_config.rbf_trainable
        rbound_upper = model_config.rbound_upper
        z_hidden_dim = model_config.z_hidden_dim
        ef_dim = model_config.ef_dim
        k_tuple_dim = model_config.k_tuple_dim
        block_num = model_config.block_num
        activation_fn_name = model_config.get("activation_fn_name", "silu")
        activation_fn = activation_fn_map[activation_fn_name]
        train_e_loss_fn = "l1"
        train_f_loss_fn = "rmse"
        metric_fn_e = "l1"
        metric_fn_f = "l1"
        
        # only for E models
        e_mode = model_config.get("e_mode")
        use_concat = model_config.get("use_concat")
        pooling_level = model_config.get("pooling_level")
        interaction_residual = model_config.get("interaction_residual", True)
        
        ema_decay = optimizer_config.ema_decay
        

        self.learning_rate = optimizer_config.learning_rate
        self.force_ratio = optimizer_config.get("force_ratio")
        self.weight_decay = optimizer_config.weight_decay
        self.gradient_clip_val = optimizer_config.gradient_clip_val
        self.automatic_optimization = False
        self.train_e_loss_fn = loss_fn_map[train_e_loss_fn]
        self.train_f_loss_fn = loss_fn_map[train_f_loss_fn]
        self.train_y_loss_fn = loss_fn_map["l2"]
        self.metric_fn_e = loss_fn_map[metric_fn_e]
        self.metric_fn_f = loss_fn_map[metric_fn_f]
        self.metric_fn_y = loss_fn_map["l1"]
        self.qm9 = qm9

        
        self.RLROP_factor = scheduler_config.RLROP_factor
        self.RLROP_patience = scheduler_config.RLROP_patience
        self.RLROP_threshold = scheduler_config.RLROP_threshold
        self.EXLR_gamma = scheduler_config.EXLR_gamma
        self.warmup_epoch = scheduler_config.warmup_epoch
        self.warmup_mult = 1.0
        self.warmup_end = False
        
        self.global_y_mean = global_y_mean
        self.global_y_std = global_y_std
        
            
        # Transform Molecule to Graph
        self.M2G = Mol2Graph(
            z_hidden_dim=z_hidden_dim,
            ef_dim=ef_dim,
            rbf=rbf,
            max_z=max_z,
            rbound_upper=rbound_upper,
            rbf_trainable=rbf_trainable
        )
        

        
        # initialize tuples
        self.init_layer = init_layer(
            z_hidden_dim=z_hidden_dim,
            ef_dim=ef_dim,
            k_tuple_dim=k_tuple_dim,
            activation_fn=activation_fn,
        )
        
        # interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(block_num):
            self.interaction_layers.append(
                    interaction_layer(
                        hidden_dim=k_tuple_dim,
                        activation_fn=activation_fn,
                        e_mode=e_mode,
                        ef_dim=ef_dim,
                        use_concat=use_concat,
                        interaction_residual=interaction_residual
                        )
                    )

        # output layers
        self.output_layers = output_layer(
            hidden_dim=k_tuple_dim,
            activation_fn=activation_fn,
            pooling_level=pooling_level
            )
        
        # ema configs
        self.ema_model = ExponentialMovingAverage(self, decay=ema_decay, device=self.device)

    def forward(self, batch):
        z, pos = batch.z, batch.pos
        
        # Molecule to Graphs
        emb1, ef = self.M2G(z, pos)  
        kemb = self.init_layer(emb1, ef)

        # interaction
        for i in range(len(self.interaction_layers)):
            kemb = self.interaction_layers[i](
                kemb=kemb,
                ef=ef
                ) + kemb

        # output
        scores = self.output_layers(
            kemb=kemb,
            pos=pos,
            z=z
            )
        
        # normalize
        return scores * self.global_y_std + self.global_y_mean
    
    def training_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]
        self.train()
        if not self.qm9:
            batch.pos.requires_grad_(True)
            
        # forward
        pred_y = self(batch).squeeze()

        # if predict_force, auto-grad for prediction, and calculate mutipile loss.
        if not self.qm9:
            pred_force = -torch.autograd.grad(
                [torch.sum(pred_y)], 
                [batch.pos],
                retain_graph=True,
                create_graph=True
                )[0]
            
            
            batch.pos.requires_grad_(False)

            pred_y = pred_y.reshape(batch.energy.shape)
            pred_force = pred_force.reshape(batch.force.shape)
            energy_loss = self.train_e_loss_fn(pred_y, batch.energy)
            force_loss = self.train_f_loss_fn(pred_force, batch.force)
            force_ratio = self.force_ratio
            loss = force_ratio * force_loss + (1 - force_ratio) * energy_loss
        
            # log multiple loss
            self.log('train_loss/train_loss_e', energy_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('train_loss/train_loss_f', force_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('train_loss/train_loss', loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=batch_size)

        # else: just single loss.
        else:
            pred_y = pred_y.reshape(batch.y.shape)
            loss = self.train_y_loss_fn(pred_y, batch.y)
            self.log('train_loss/train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=batch_size)
            
            
        # optimize manually
        lr = self.lr_schedulers()[0].optimizer.param_groups[0]["lr"]
        self.optimizers().optimizer.param_groups[0]["lr"] = lr
        self.optimizers().param_groups[0]["lr"] = lr
        
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        clip_grad_norm_(self.parameters(), self.gradient_clip_val)
        self.optimizers().step()
                
        
        # log
        self.log("learning_rate/lr_rate_AdamW", self.optimizers().param_groups[0]["lr"], on_epoch=True, on_step=False, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        batch_size = batch.z.shape[0]
        
        # if predict_force, auto-grad for prediction, and calculate mutipile loss.
        if not self.qm9:
            batch.pos.requires_grad_(True)
            
        
        # forward
        self.eval()
        pred_y = self.ema_model(batch).squeeze()
        
        if not self.qm9:
            pred_force = -torch.autograd.grad(
                [torch.sum(pred_y)], 
                [batch.pos],
                retain_graph=False,
                create_graph=False
                )[0]
            
            
            batch.pos.requires_grad_(False)
        
        
        # if predict_force, calculate mutipile loss.
        if not self.qm9:
            force_ratio = self.force_ratio
            
            pred_y = pred_y.reshape(batch.energy.shape)
            pred_force = pred_force.reshape(batch.force.shape)
            
            energy_loss = self.metric_fn_e(pred_y, batch.energy)
            force_loss = self.metric_fn_f(pred_force, batch.force)
            loss = force_ratio * force_loss + (1 - force_ratio) * energy_loss
            
            # log multiple loss
            self.log('val_loss/val_loss_e', energy_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('val_loss/val_loss_f', force_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('val_loss/val_loss', loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=batch_size)
        # validation loss for single prediction
        else:
            pred_y = pred_y.reshape(batch.y.shape)
            loss = self.metric_fn_y(pred_y, batch.y)
            self.log('val_loss/val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        
        
            
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]
        with torch.inference_mode(False):
            # clone tensor (origin tensor are in inference mode)
            batch.z = batch.z.clone()
            batch.pos = batch.pos.clone()
            if not self.qm9:
                batch.energy = batch.energy.clone()
                batch.force = batch.force.clone()
                batch.pos.requires_grad_(True)
            else:
                batch.y = batch.y.clone()

            # forward
            self.eval()
            pred_y = self.ema_model(batch).squeeze()
            

                
            log_prefix = self.test_dataset_name + "/" if self.test_dataset_name is not None else ""

                
            # if predict_force, auto-grad for prediction, and calculate mutipile loss.
            if not self.qm9:
                pred_force = -torch.autograd.grad(
                    [torch.sum(pred_y)], 
                    [batch.pos],
                    retain_graph=False,
                    create_graph=False
                    )[0]
                batch.pos.requires_grad_(False)


                pred_y = pred_y.reshape(batch.energy.shape)
                pred_force = pred_force.reshape(batch.force.shape)
                
                energy_loss = self.metric_fn_e(pred_y, batch.energy)
                force_loss = self.metric_fn_f(pred_force, batch.force)
                
                
                self.log(f'test_loss/{log_prefix}test_loss_e', energy_loss, batch_size=batch_size)
                self.log(f'test_loss/{log_prefix}test_loss_f', force_loss, batch_size=batch_size)
                
                return energy_loss, force_loss
            # test loss for single prediction
            else:
                pred_y = pred_y.reshape(batch.y.shape)
                
                loss = self.metric_fn_y(pred_y, batch.y)
                self.log(f'test_loss/{log_prefix}test_loss', loss, batch_size=batch_size)

                return loss
            
    def configure_optimizers(self):
        # initialize AdamW
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-07,
            amsgrad=True
            )

        # initialize schedulers
        RLROP = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=self.RLROP_factor,
            patience=self.RLROP_patience,
            threshold=self.RLROP_threshold,
        )
        
        EXLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.EXLR_gamma)
        EXLR_warmup = GradualWarmupScheduler(
            optimizer=optimizer, 
            multiplier=self.warmup_mult, 
            total_epoch=self.warmup_epoch, 
            after_scheduler=EXLR
            )
        
        lr_scheduler_configs = []
        for sched in [RLROP, EXLR_warmup]:
            lr_scheduler_config = {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val_loss/val_loss"
            }
            lr_scheduler_configs.append(lr_scheduler_config)


        return [optimizer], lr_scheduler_configs
            


