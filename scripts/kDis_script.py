import torch
import sys, os
os.chdir("/home/lizian/codes/DisGNN_github")
sys.path.append(".")
import pytorch_lightning as pl
import time
from argparse import ArgumentParser
from scripts.script_utils import trainer_setup, test, train, get_cfgs
from utils.select_free_gpu import select_free_gpu
from lightningModule.MD_module import MD_module
from datasets.MD17 import md17_datawork
from datasets.QM9 import qm9_datawork

'''
    get args
'''
parser = ArgumentParser()
parser.add_argument("--model", choices=["2EDis", "2FDis", "3EDis"], default="2FDis")
parser.add_argument("--ds", choices=["qm9", "md17"], default="md17")
parser.add_argument("--dname", default="ethanol")
parser.add_argument("--devices", nargs="+", type=int, default=None)
parser.add_argument("--data_dir", default="~/datasets/MD17")
parser.add_argument("--version", default="NO_VERSION")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--only_test", action="store_true")
parser.add_argument("--ckpt", default=None)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--proj_name", default=None)
parser.add_argument("--merge", nargs="+", type=str, default=None)


args = parser.parse_args()
# print log
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

model_name = args.model
dataset_name = args.ds
data_name = args.dname
data_dir = args.data_dir
resume = args.resume
checkpoint_path = args.ckpt
only_test = args.only_test
version = args.version
devices = args.devices
merge_list = args.merge
accelerator = "gpu"
if devices is None:
    devices = [select_free_gpu()]
elif devices == [-1]:
    devices = None
    accelerator = "cpu"


'''
    get hparams
'''
config_path = "hparams/{}_{}.yaml".format(model_name, dataset_name)
specific_config_path = "hparams/specific/{}_{}_specific.yaml".format(model_name, dataset_name)
if not os.path.exists(specific_config_path):
    specific_config_path = None
config = get_cfgs(config_path, merge_list, specific_config_path, data_name)

print("-"*20)
print(config)
print("-"*20)

scheduler_config = config.scheduler_config
optimizer_config = config.optimizer_config
model_config = config.model_config

# trainer_config
trainer_config = config.trainer_config
validation_interval = trainer_config.validation_interval
log_every_n_steps = 10
early_stopping_patience = trainer_config.early_stopping_patience
max_epochs = trainer_config.max_epochs

# global_config
global_config = config.global_config
seed = config.global_config.seed

# data_config
data_config = config.data_config
train_batch_size = data_config.train_batch_size
val_batch_size = data_config.val_batch_size
test_batch_size = data_config.test_batch_size
    
    

    
'''
    get model class
'''

model = MD_module

'''
    train start
'''
    
pl.seed_everything(seed)

'''
    prepare data
'''

data_work_dict = {
    "qm9": qm9_datawork,
    "md17": md17_datawork,
}

datawork = data_work_dict[dataset_name]



train_dl, val_dl, test_dl, global_y_mean, global_y_std = datawork(
    name=data_name,
    root=data_dir,
    batch_size=[train_batch_size, val_batch_size, test_batch_size],
)

    
'''
    prepare model
'''

if only_test or resume and checkpoint_path is not None:
    model_instance = model.load_from_checkpoint(
        checkpoint_path=checkpoint_path
        )
else:
    if (only_test or resume):
        print("WARNING: You are resuming but not specifying any ckpt.")
    model_instance = model(
        model_name=model_name,
        model_config=model_config, 
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        global_y_std=global_y_std, 
        global_y_mean=global_y_mean,
        qm9=(dataset_name == "qm9"),
        data_name=data_name
    )


'''
    prepare trainer
'''


log_path = "{}_log/{}/{}".format(
    model_name, 
    dataset_name, 
    data_name
    ) 

trainer = trainer_setup(
    log_path=log_path,
    version=version,
    early_stopping_patience=early_stopping_patience,
    max_epochs=max_epochs,
    validation_interval=validation_interval,
    devices=devices,
    log_every_n_steps=log_every_n_steps,
    accelerator=accelerator,
    use_wandb=args.use_wandb,
    proj_name=args.proj_name if args.proj_name is not None else f"{model_name}_{dataset_name}",
    data_name=data_name,
)


'''
    train and test
'''

if not only_test:
    start_time = time.time()
    train(
        trainer=trainer,
        model=model_instance,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        resume=resume,
        ckpt_path=checkpoint_path,
    )
    end_time = time.time()
    

test_loss = test(
    trainer=trainer,
    model=model_instance,
    test_dataloader=test_dl,
    only_test=only_test,
    ckpt_path=checkpoint_path
)
