import torch
import subprocess

def get_gpu_info():
    try:
        out_str = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'], universal_newlines=True)
        out_list = out_str.strip().split('\n')[1:]
        used_memory = torch.tensor([int(x.strip().split(', ')[0][:-4]) for x in out_list])
        return used_memory
    except:
        return None

def select_free_gpu():
    gpu_info = get_gpu_info()
    assert gpu_info is not None
    gpu_idx = int(gpu_info.argmin().item())
    return gpu_idx
