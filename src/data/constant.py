import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image

class ConstantDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 64, data_dir: str = "./data", seed: int = 42, num_workers: int = 16):
        super().__init__()

        self.batch_size = batch_size
        self.dims = (1, 28, 28)
        self.num_workers = num_workers
      
        transform = []
        transform.append(transforms.ToTensor())
            
        self.transform = transforms.Compose(transform)

    def prepare_data(self):
        pass

    def setup(self):
        self.train_dataset = ConstantDataset(value=0, shape=(55000, 1, 28, 28))
        self.val_dataset = ConstantDataset(value=0, shape=(5000, 1, 28, 28))
        self.test_dataset = ConstantDataset(value=0, shape=(10000, 1, 28, 28))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)    

class ConstantDataset(torch.utils.data.Dataset):
    def __init__(self, value=0, shape=(10000, 1, 28, 28)):

        self.values = torch.Tensor(np.zeros(shape) + value).float()
        self.labels = np.zeros(shape[0])

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, shape=(10000, 1, 28, 28)):

        self.values = torch.Tensor(np.random.uniform(0, 1, shape)).float()
        self.labels = np.zeros(shape[0])

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]
