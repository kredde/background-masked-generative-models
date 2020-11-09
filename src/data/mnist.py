import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 64, data_dir: str = "./data", seed: int = 42, num_workers: int = 16, normalize: bool = False):
        super().__init__()

        self.batch_size = batch_size
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.seed = seed

        if normalize:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # download only
        MNIST(self.data_dir, train=True, download=True,
              transform=self.transform)
        MNIST(self.data_dir, train=False, download=True,
              transform=self.transform)

    def setup(self):
        # transform
        mnist_train = MNIST(self.data_dir, train=True,
                            download=False, transform=self.transform)
        mnist_test = MNIST(self.data_dir, train=False,
                           download=False, transform=self.transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)
