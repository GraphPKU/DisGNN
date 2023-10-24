import pytorch_lightning as pl
import sys
import os
from argparse import ArgumentParser
sys.path.append(".")
from scripts.script_utils import get_cfgs
from utils.select_free_gpu import select_free_gpu
from lightningModule.MD_module import MD_module
from datasets.QM9 import qm9_datawork

'''
    get args
'''
parser = ArgumentParser()
parser.add_argument("--model", choices=["2FDis"], default="2FDis")
parser.add_argument("--devices", nargs="+", type=int, default=None)
parser.add_argument("--data_dir", default="~/datasets/QM9")
parser.add_argument("--version", default="NO_VERSION")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--only_test", action="store_true")
parser.add_argument("--ckpt", default=None)
parser.add_argument("--merge", nargs="+", type=str, default=None)




args = parser.parse_args()
# print log
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

model_name = args.model
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
config_path = "hparams/{}_{}.yaml".format(model_name, "qm9")
specific_config_path = "hparams/specific/{}_{}_specific.yaml".format(model_name, "qm9")
if not os.path.exists(specific_config_path):
    specific_config_path = None
config = get_cfgs(config_path, merge_list, specific_config_path, "2")

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

    
model = MD_module
    
pl.seed_everything(seed)





train_dl, val_dl, test_dl, global_y_mean, global_y_std = qm9_datawork(
    name="4",
    root=data_dir,
    batch_size=[16, 16, 16],
)

    
'''
    prepare model
'''
# specify the checkpoints for HOMO and LUMO targets
checkpoint_path1 = ''
checkpoint_path2 = ''
device = "cuda:0"

model_instance1 = model.load_from_checkpoint(
    checkpoint_path=checkpoint_path1,
    map_location=device
    )

model_instance1.to(device)

model_instance2 = model.load_from_checkpoint(
    checkpoint_path=checkpoint_path2,
    map_location=device
    
    )
model_instance2.to(device)


'''
    train and test
'''

# only_test

all_num = 0
sum_loss = 0
from tqdm import tqdm
for data in tqdm(iter(test_dl)):
    data.to(device)
    
    out = - model_instance1.ema_model(data) + model_instance2.ema_model(data)
    
    label = data.y
    from torch.nn import functional as F
    
    # shape as label
    out = out.view(label.shape)
    
    loss = F.l1_loss(out, label)
    
    batch_num = data.z.shape[0]
    all_num += batch_num
    
    sum_loss += loss.item() * batch_num
    
print(sum_loss / all_num)