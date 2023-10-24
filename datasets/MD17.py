from typing import List, Optional
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch_geometric.datasets import md17
from typing import Optional, List, Callable
from torch_geometric.data import Data
import torch
import numpy as np

# preprocess work
def collate(batch):
    return Data(
        pos = torch.cat([data.pos.unsqueeze(0) for data in batch], dim=0),
        z = torch.cat([data.z.unsqueeze(0) for data in batch], dim=0),
        energy = torch.cat([data.energy for data in batch], dim=0),
        force = torch.cat([data.force.unsqueeze(0) for data in batch], dim=0),
    )


class my_MD17(md17.MD17):
    def __init__(
        self,
        root: str,
        name: str,
        train: Optional[bool] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, name, train, transform, pre_transform, pre_filter)
        
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    def process(self):
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)

            if self.revised:
                z = torch.from_numpy(raw_data['nuclear_charges']).long()
                pos = torch.from_numpy(raw_data['coords']).float()
                energy = torch.from_numpy(raw_data['energies']).float()
                force = torch.from_numpy(raw_data['forces']).float()
            else:
                z = torch.from_numpy(raw_data['z']).long()
                pos = torch.from_numpy(raw_data['R']).float()
                energy = torch.from_numpy(raw_data['E']).float()
                force = torch.from_numpy(raw_data['F']).float()

            data_list = []
            import tqdm
            for i in tqdm.tqdm(range(pos.size(0))):
                data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            torch.save(self.collate(data_list), processed_path)
    
    
def md17_datawork(
    root: str, 
    name: str,
    batch_size: List[int],
    **kwargs
    ):
    revised = 'revised' in name

    data_point_num = [950, 50] if revised else [1000, 1000] 
    
    # get dataset and collate function.
    dataset = my_MD17(root=root, name=name)
    # get meta data
    global_y_mean = dataset.data.energy.mean()
    global_y_std = dataset.data.energy.std()

    # get basic configs
    train_data_num, val_data_num = data_point_num
    test_data_num = len(dataset) - train_data_num - val_data_num
    train_batch_size, val_batch_size, test_batch_size = batch_size
    
    # random_split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_data_num, 
        val_data_num, 
        test_data_num]
        )
    
    
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = (
        DataLoader(train_dataset, num_workers=8, batch_size=train_batch_size, persistent_workers=True, shuffle=True, collate_fn=collate),
        DataLoader(val_dataset, num_workers=4, batch_size=val_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate),
        DataLoader(test_dataset, num_workers=4, batch_size=test_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate),
    )
    
    return train_dataloader, val_dataloader, test_dataloader, global_y_mean, global_y_std
