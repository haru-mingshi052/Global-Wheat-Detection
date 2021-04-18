import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import create_data
from .transformer import *

"""
DataLoaderの作成
"""

#==============================
# GWDDataset
#==============================
class GWDDataset(Dataset):
    def __init__(self, df, image_dir, transforms):
        super().__init__()
        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 225.0
        
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        area = torch.as_tensor(area, dtype = torch.float32)
        
        labels = torch.ones((records.shape[0],), dtype = torch.int64)
        
        iscrowd = torch.zeros((records.shape[0],), dtype = torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms:
            sample = {
                'image' : image,
                'bboxes' : target['boxes'],
                'labels' : labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)
            
        return image, target, image_id
    
    def __len__(self):
        return self.image_ids.shape[0]

#=================================
# GWS_testDataset
#=================================
class GWD_testDataset(Dataset):
    def __init__(self, df, image_dir, transforms):
        super().__init__()
        
        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 225.0
        
        if self.transforms:
            sample = {
                'image' : image,
            }
            
            sample = self.transforms(**sample)
            image = sample['image']
            
        return image, image_id
    
    def __len__(self):
        return self.image_ids.shape[0]

#=========================
# collate_fn
#=========================
def collate_fn(batch):
    return tuple(zip(*batch))

#=======================================
# create_dataloader
#=======================================
def create_dataloader(data_folder):
    train, val, test = create_data(data_folder)
    
    train_ds = GWDDataset(train, data_folder + '/train', train_transform())
    val_ds = GWDDataset(val, data_folder + '/train', val_transform())
    test_ds = GWD_testDataset(test, data_folder + '/test', test_transform())

    tr_dl = DataLoader(train_ds, batch_size = 16, shuffle = True, collate_fn = collate_fn)
    val_dl = DataLoader(val_ds, batch_size = 16, shuffle = True, collate_fn = collate_fn)
    test_dl = DataLoader(test_ds, batch_size = 16, shuffle = False, collate_fn = collate_fn)

    return tr_dl, val_dl, test_dl