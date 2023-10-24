from torch_geometric.datasets import qm9
from torch.utils.data import random_split
from typing import Optional, List
from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Data
from typing import Callable
import random

def collate_(batch, y_index=0):
    # avoid forming the batch indice.
    data = batch[0]
    try:
        data.y = data.y.squeeze(dim=1)[:, y_index]
    except:
        pass
    return data



def group_same_size(
    dataset: Data,
):
    data_list = list(dataset)
    data_list.sort(key=lambda data: data.z.shape[0])
    # grouped dataset by size
    grouped_dataset = []
    for i in range(len(data_list)):
        data = data_list[i]
        if i == 0:
            group = [data]
        else:
            last_data = data_list[i-1]
            if data.z.shape[0] == last_data.z.shape[0]:
                group.append(data)
            else:
                grouped_dataset.append((last_data.z.shape[0], group))
                group = [data]
    return grouped_dataset

def batch_same_size(
    grouped_dataset: Data,
    batch_size: int,
):
    # batched dataset, according to the batch size. 
    batched_dataset = []
    for size, group in grouped_dataset:
        batch_num_in_group = (len(group) // batch_size) + 1 if len(group) % batch_size != 0 else len(group) // batch_size
        for i in range(batch_num_in_group):
            lower_bound = i * batch_size
            upper_bound = min((i+1) * batch_size, len(group))
            
            batch = group[lower_bound:upper_bound]
            y = torch.cat([batch[i].y.unsqueeze(0) for i in range(len(batch))], dim=0)
            z = torch.cat([batch[i].z.unsqueeze(0) for i in range(len(batch))], dim=0)
            pos = torch.cat([batch[i].pos.unsqueeze(0) for i in range(len(batch))], dim=0)
            batched_dataset.append(
                Data(
                    y=y, 
                    z=z, 
                    pos=pos,
                    batch_size=z.shape[0],
                    graph_size=z.shape[1],
                )
            )
    return batched_dataset

class batched_QM9(qm9.QM9):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                    pre_transform: Optional[Callable] = None,
                    pre_filter: Optional[Callable] = None,
                    batch_size: int = None,
                    indices: List[int] = None):
        self.size = len(indices)
        self.batch_size = batch_size
        self.flag = False
        super().__init__(root, transform, pre_transform, pre_filter)
        # To batch train/val/test data respectively. For using, please set the indices of the data.
        self.subdataset = self[indices]
        self.grouped_data = group_same_size(self.subdataset)
        self.batched_data = batch_same_size(self.grouped_data, self.batch_size)
        self.flag = True

        self.data_num = self.size
        self.size = len(self.batched_data) # batched version, a datapoint is a batch.

    def __getitem__(self, index):
        # to get the whole batched data.
        if not self.flag:
            return super().__getitem__(index)
        return self.batched_data[index]
    def __len__(self):
        return self.size
    
    def __repr__(self) -> str:
        return f"Batched_QM9(batch_size={self.batch_size}, size={self.size})"
    
    def process(self):
        data_list = torch.load(self.raw_paths[0])
        data_list = [Data(**data_dict) for data_dict in data_list]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            import tqdm
            data_list = [self.pre_transform(d) for (d) in tqdm.tqdm(data_list)]
        torch.save(self.collate(data_list), self.processed_paths[0])

    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def reshuffle_grouped_dataset(self):
        for _, group in self.grouped_data:
            random.shuffle(group)
        self.batched_data = batch_same_size(self.grouped_data, self.batch_size)


def qm9_datawork(
    name: str,
    root: str,
    batch_size: List[int],
    ):
    name = name if type(name) == int else int(name)
    
    # get dataset and collate function.
    from functools import partial
    QM9_dataset_group_partial = partial(batched_QM9, root=root)
    
    # get basic configs
    all_data_num = 130831
    data_point_num = [110000, 10000]
    train_data_num, val_data_num = data_point_num
    test_data_num = all_data_num - train_data_num - val_data_num
    
    # random_split indices according to data_point_num
    train_indices, val_indices, test_indices = random_split(range(all_data_num), [train_data_num, val_data_num, test_data_num]) 
    train_indices, val_indices, test_indices = list(train_indices), list(val_indices), list(test_indices)
    train_batch_size, val_batch_size, test_batch_size = batch_size
    
    collate = lambda data_list: collate_(data_list, name)
    
    train_dataset, val_dataset, test_dataset = (
        QM9_dataset_group_partial(indices=train_indices, batch_size=train_batch_size),
        QM9_dataset_group_partial(indices=val_indices, batch_size=val_batch_size),
        QM9_dataset_group_partial(indices=test_indices, batch_size=test_batch_size),
    )
    
    # get dataloaders, note that batch size must be 1 because batch is already divided in dataset.
    train_dataloader, val_dataloader, test_dataloader = (
        DataLoader(train_dataset, num_workers=4, batch_size=1, persistent_workers=False, shuffle=True, collate_fn=collate), 
        DataLoader(val_dataset, num_workers=4, batch_size=1, persistent_workers=False, shuffle=False, collate_fn=collate), 
        DataLoader(test_dataset, num_workers=4, batch_size=1, persistent_workers=False, shuffle=False, collate_fn=collate),
    )
    
    # calculate global mean and std
    all_dataset = train_dataset + val_dataset + test_dataset
    global_y_mean = torch.cat([all_dataset[i].y for i in range(len(all_dataset))], dim=0).mean(dim=0).squeeze()[name]
    global_y_std = torch.cat([all_dataset[i].y for i in range(len(all_dataset))], dim=0).std(dim=0).squeeze()[name]
    
    return train_dataloader, val_dataloader, test_dataloader, global_y_mean, global_y_std


