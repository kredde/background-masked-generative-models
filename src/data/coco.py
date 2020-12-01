import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from math import floor
import sys

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class COCODataModule(LightningDataModule):
    
    def __init__(self, batch_size: int = 64, foreground_data_dir: str = "./data/COCO/foreground_images/", background_data_dir: str = "./data/COCO/background_images/",seed: int = 42, num_workers: int = 8, 
    normalize: bool = False, convert_grayscale: bool = False, split_ratio: float = 0.8):
        
        super().__init__()
        
        self.batch_size = batch_size        
        self.foreground_data_dir = foreground_data_dir
        self.background_data_dir = background_data_dir
        self.normalize = normalize
        self.convert_grayscale = convert_grayscale
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.seed = seed
        
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32, 32), interpolation=Image.BICUBIC), transforms.ToTensor()]) if convert_grayscale else transforms.Compose([transforms.Resize((32, 32), interpolation=Image.BICUBIC), transforms.ToTensor()])

        
    def prepare_data(self):  
        self.foreground_trainvaltest = datasets.ImageFolder(self.foreground_data_dir, transform=self.transform)
        self.background_trainvaltest = datasets.ImageFolder(self.background_data_dir, transform=self.transform)
    
    
    def setup(self):
        train_data_len = floor((len(self.foreground_trainvaltest) * self.split_ratio))
        val_data_len = floor(len(self.foreground_trainvaltest) * ((1 - self.split_ratio) * 0.5))
        test_data_len = floor(len(self.foreground_trainvaltest) * ((1 - self.split_ratio) * 0.5))
        
        redundant = len(self.foreground_trainvaltest) - train_data_len - val_data_len - test_data_len
        train_data_len += redundant 
        
        #train/val/test split. Seed value has been set to same value in order to keep the order btw foreground and background 
        self.foreground_train, self.foreground_val, self.foreground_test = random_split(self.foreground_trainvaltest, [train_data_len, val_data_len, test_data_len], generator=torch.Generator().manual_seed(self.seed))

        self.background_train, self.background_val, self.background_test = random_split(self.background_trainvaltest, [train_data_len, val_data_len, test_data_len], generator=torch.Generator().manual_seed(self.seed))
    
    
    def train_dataloader(self):
        concat_dataset = ConcatDataset(
            self.background_train,
            self.foreground_train
        )
        return DataLoader(concat_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        concat_dataset = ConcatDataset(
            self.background_val,
            self.foreground_val
        )
        return DataLoader(concat_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)
    
    def test_dataloader(self):
        concat_dataset = ConcatDataset(
            self.background_test,
            self.foreground_test
        )
        return DataLoader(concat_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)