import pytorch_lightning as pl
import torch.nn as nn
from layers.Mol2Graph import Mol2Graph
from utils.loss_fns import loss_fn_map
from utils.activation_fns import activation_fn_map
from utils.EMA import ExponentialMovingAverage
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch

from layers.kDisGNN_model import kDisGNN

from layers.basic_layers import Residual, Dense
from utils.GradualWarmupScheduler import GradualWarmupScheduler




class MD_module(pl.LightningModule):
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
        
        '''
            model configs
        '''
        
        self.model = kDisGNN(
            z_hidden_dim=model_config.z_hidden_dim,
            ef_dim=model_config.ef_dim,
            rbf=model_config.rbf,
            max_z=model_config.max_z,
            rbound_upper=model_config.rbound_upper,
            rbf_trainable=model_config.rbf_trainable,
            activation_fn=activation_fn_map[model_config.activation_fn_name],
            k_tuple_dim=model_config.k_tuple_dim,
            block_num=model_config.block_num,
            pooling_level=model_config.get("pooling_level"),
            e_mode=model_config.get("e_mode"),
            qm9=qm9,
            model_name=model_name,
            use_mult_lin=model_config.get("use_mult_lin"),
            data_name=kwargs.get("data_name"),
            interaction_residual=model_config.get("interaction_residual"),
            global_y_mean=global_y_mean,
            global_y_std=global_y_std,
            )

        
        ema_decay = optimizer_config.ema_decay
        

        self.learning_rate = optimizer_config.learning_rate
        self.force_ratio = optimizer_config.get("force_ratio")
        self.weight_decay = optimizer_config.weight_decay
        self.gradient_clip_val = optimizer_config.gradient_clip_val
        self.automatic_optimization = False
        self.train_e_loss_fn = loss_fn_map["l1"]
        self.train_f_loss_fn = loss_fn_map["rmse"]
        self.train_y_loss_fn = loss_fn_map["l2"]
        self.metric_fn_e = loss_fn_map["l1"]
        self.metric_fn_f = loss_fn_map["l1"]
        self.metric_fn_y = loss_fn_map["l1"]
        self.qm9 = qm9

        
        self.RLROP_factor = scheduler_config.RLROP_factor
        self.RLROP_patience = scheduler_config.RLROP_patience
        self.RLROP_threshold = scheduler_config.RLROP_threshold
        self.EXLR_gamma = scheduler_config.EXLR_gamma        
        self.RLROP_cooldown = scheduler_config.RLROP_cooldown

        self.warmup_epoch = scheduler_config.warmup_epoch
        self.warmup_mult = 1.0
        self.warmup_end = False
        
        # ema configs
        self.ema_model = ExponentialMovingAverage(self, decay=ema_decay, device=self.device)

    def forward(self, batch_data):
        return self.model(batch_data)
    
    def training_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]
        self.train()
            
        # forward
        step_func = self.qm9_step if self.qm9 else self.molsim_step
        loss = step_func(batch, period="train")
        
        if loss != loss:
            print("nan loss, skippping")
            return loss
            
        # optimize manually
        self.opt_step(loss)
        # log
        self.log("learning_rate/lr_rate_AdamW", self.optimizers().param_groups[0]["lr"], on_epoch=True, on_step=False, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.eval()
        
        # forward
        step_func = self.qm9_step if self.qm9 else self.molsim_step
        loss = step_func(batch, period="val")
            
        
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]
        with torch.inference_mode(False):
            # clone tensor (origin tensor are in inference mode)
            for key in batch.keys:
                try:
                    batch[key] = batch[key].clone()
                except:
                    pass

            # forward
            self.eval()
            step_func = self.qm9_step if self.qm9 else self.molsim_step
            return step_func(batch, period="test")
            
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
            cooldown=self.RLROP_cooldown,
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
            


    def opt_step(self, loss):
        lr = self.lr_schedulers()[0].optimizer.param_groups[0]["lr"]
        self.optimizers().optimizer.param_groups[0]["lr"] = lr
        self.optimizers().param_groups[0]["lr"] = lr
        
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        clip_grad_norm_(self.parameters(), self.gradient_clip_val)
        self.optimizers().step()
        
        
    def molsim_step(self, batch, period=None):
        
        train = period == "train"
        
        model = self if train else self.ema_model
        e_loss_fn = self.train_e_loss_fn if train else self.metric_fn_e
        f_loss_fn = self.train_f_loss_fn if train else self.metric_fn_f
        
        
        pred_y, pred_force = model(batch)
        pred_y = pred_y.squeeze()
        
        pred_y = pred_y.reshape(batch.energy.shape)
        pred_force = pred_force.reshape(batch.force.shape)
        energy_loss = e_loss_fn(pred_y, batch.energy)
        force_loss = f_loss_fn(pred_force, batch.force)
        loss = self.force_ratio * force_loss + (1 - self.force_ratio) * energy_loss
    
        # log multiple loss
        batch_size = batch.z.shape[0]
        
        if period == "train":
            self.log('train_loss/train_loss_e', energy_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('train_loss/train_loss_f', force_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('train_loss/train_loss', loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=batch_size)
            return loss

        elif period == "val":
            self.log('val_loss/val_loss_e', energy_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('val_loss/val_loss_f', force_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
            self.log('val_loss/val_loss', loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=batch_size)
            return loss
            
        else:
            self.log(f'test_loss/test_loss_e', energy_loss, batch_size=batch_size)
            self.log(f'test_loss/test_loss_f', force_loss, batch_size=batch_size)
            return energy_loss, force_loss
        
            
    def qm9_step(self, batch, period=None):
        
        train = period == "train"
        model = self if train else self.ema_model
        loss_fn = self.train_y_loss_fn if train else self.metric_fn_y
        
        
        
        pred_y = model(batch)
        pred_y = pred_y.reshape(batch.y.shape)
        
        loss = loss_fn(pred_y, batch.y)
        batch_size = batch.z.shape[0]
        
        if period == "train":
            self.log('train_loss/train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=batch_size)
        elif period == "val":
            self.log('val_loss/val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        else:
            self.log(f'test_loss/test_loss', loss, batch_size=batch_size)
        
        return loss        
        