import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from PIL import Image

from src.utils.pixelcnn import randomize_background
from src.data.mnist import MNISTDataModule
from src.data.fashionmnist import FashionMNISTDataModule

from src.utils.vae import randomize_background as randomize_background_vae


class MNISTBgAug(MNIST):
    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img1 = torch.clone(img)
        img2 = torch.clone(img)

        img1 = randomize_background(img1, norm=1)
        img2 = randomize_background(img2, norm=1)

        return (img, img1, img2), target

class MNISTBgAugSingleImg(MNIST):
    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        mask = img.clone()
        img = randomize_background_vae(img)

        return img, mask, target


class FashionMNISTBgAug(FashionMNIST):

    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img1 = torch.clone(img)
        img2 = torch.clone(img)

        img1 = randomize_background(img1, norm=1)
        img2 = randomize_background(img2, norm=1)

        return img, img1, img2, target


class FashionMNISTBgAugSingleImg(FashionMNIST):
    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        mask = img.clone()
        img = randomize_background_vae(img)

        return img, mask, target



class BgAugMNISTDataModule(MNISTDataModule):
    def __init__(self, single_image: bool = False):
        super(BgAugMNISTDataModule, self).__init__()

        self.single_image = single_image

    def prepare_data(self):
        # download only
        if not self.single_image:
            MNISTBgAug(self.data_dir, train=True, download=True,
            transform=self.transform)

            MNISTBgAug(self.data_dir, train=False, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]))
        else:
            MNISTBgAugSingleImg(self.data_dir, train=True, download=True,
            transform=self.transform)

            MNISTBgAugSingleImg(self.data_dir, train=False, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]))

    def setup(self):
        # transform
        if not self.single_image:
            mnist_train = MNISTBgAug(self.data_dir, train=True,
            download=False, transform=self.transform)
            
            mnist_test = MNISTBgAug(self.data_dir, train=False,
            download=False, transform=self.transform)
        
        else:
            mnist_train = MNISTBgAugSingleImg(self.data_dir, train=True,
            download=False, transform=self.transform)

            mnist_test = MNISTBgAugSingleImg(self.data_dir, train=False,
            download=False, transform=self.transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test


class BgAugFashionMNISTDataModule(FashionMNISTDataModule):
    def __init__(self, single_image: bool = False):
        super(BgAugFashionMNISTDataModule, self).__init__()

        self.single_image = single_image

    def prepare_data(self):

        if not self.single_image:
        # download only
            FashionMNISTBgAug(self.data_dir, train=True, download=True,
            transform=self.transform)

            FashionMNISTBgAug(self.data_dir, train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
        
        else:
            FashionMNISTBgAugSingleImg(self.data_dir, train=True, download=True,
            transform=self.transform)

            FashionMNISTBgAugSingleImg(self.data_dir, train=False, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]))

    def setup(self):
        # transform
        if not self.single_image:
            mnist_train = FashionMNISTBgAug(self.data_dir, train=True,
            download=False, transform=self.transform)
            
            mnist_test = FashionMNISTBgAug(self.data_dir, train=False,
            download=False, transform=self.transform)
        
        else:
            mnist_train = FashionMNISTBgAugSingleImg(self.data_dir, train=True,
            download=False, transform=self.transform)

            mnist_test = FashionMNISTBgAugSingleImg(self.data_dir, train=False,
            download=False, transform=self.transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test
