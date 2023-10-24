import torch
from pytorch_lightning import Callback
from utils.GradualWarmupScheduler import GradualWarmupScheduler



class basic_train_callback(Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.ema_model.update_parameters(pl_module) 
        


    def on_train_epoch_end(self, trainer, pl_module):
        
        if pl_module.warmup_end == False:
            if pl_module.current_epoch >= pl_module.warmup_epoch:
                pl_module.warmup_end = True
            
        for scheduler in pl_module.lr_schedulers():
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

                lr = scheduler.get_last_lr()[0]
                pl_module.optimizers().optimizer.param_groups[0]["lr"] = lr
                pl_module.optimizers().param_groups[0]["lr"] = lr
        
        if pl_module.qm9:
            pl_module.trainer.train_dataloader.dataset.reshuffle_grouped_dataset()


    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.warmup_end:
            for scheduler in pl_module.lr_schedulers():
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(pl_module.trainer.callback_metrics["val_loss/val_loss"])
                    
                    lr = scheduler.optimizer.param_groups[0]["lr"]
                    pl_module.optimizers().optimizer.param_groups[0]["lr"] = lr
                    pl_module.optimizers().param_groups[0]["lr"] = lr
                    pl_module.lr_schedulers()[1].after_scheduler.optimizer.param_groups[0]["lr"] = lr
                    