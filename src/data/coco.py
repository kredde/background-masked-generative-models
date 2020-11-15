import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np

class COCODataModule(LightningDataModule):
    
    def __init__(self, batch_size: int = 64, foreground_data_dir: str = "./data/COCO/foreground_images/", background_data_dir: str = "./data/COCO/background_images/",seed: int = 42, num_workers: int = 16, normalize: bool = False):
        
        super().__init__()
        
        self.batch_size = batch_size
#         self.dims = (1, 28, 28) check the dimensions of images
        
        self.foreground_data_dir = foreground_data_dir
        self.background_data_dir = background_data_dir
        self.num_workers = num_workers
        self.seed = seed

        if normalize:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            

    def prepare_data(self):  
        foreground_trainval = datasets.ImageFolder(self.foreground_data_dir)
        background_trainval = datasets.ImageFolder(self.background_data_dir)
        
        #train/val split. Seed value has been set to same value in order to keep the order btw foreground and background 
        self.foreground_train, self.foreground_val = random_split(foreground_trainval, [50588, 12647], generator=torch.Generator().manual_seed(42))
        
        self.background_train, self.background_val = random_split(background_trainval, [50588, 12647], generator=torch.Generator().manual_seed(42))
        
        return self.foreground_train, self.foreground_val, self.background_train, self.background_val
    
    def train_dataloader(self):
        return [DataLoader(self.background_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True), DataLoader(self.foreground_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)]

    def val_dataloader(self):
        return [DataLoader(self.background_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True), DataLoader(self.foreground_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)]