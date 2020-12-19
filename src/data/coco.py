import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from math import floor
import sys
from src.utils.pixelcnn import randomize_background, randomize_background_normal


class ConcatDataset(Dataset):
    def __init__(self, background, foreground, rand_bg: bool = False, rand_normal_bg: bool = False, bg_aug_max: float = 0.5):
        self.bg = background
        self.fg = foreground
        self.rand_bg = rand_bg
        self.rand_normal_bg = rand_normal_bg
        self.bg_aug_max = bg_aug_max

    def __getitem__(self, i):
        bg = self.bg[i]
        fg_img, fg_targ = self.fg[i]

        if self.rand_bg:
            if self.rand_normal_bg:
                fg_rand_img = randomize_background_normal(fg_img)
            else:
                fg_rand_img = randomize_background(
                    fg_img, norm=self.bg_aug_max)
            return (bg, (fg_img, fg_rand_img, fg_targ))

        return (self.bg[i], self.fg[i])

    def __len__(self):
        return min(len(d) for d in [self.bg, self.fg])


class COCODataModule(LightningDataModule):

    def __init__(self, batch_size: int = 64, foreground_data_dir: str = "./data/COCO/foreground_images/", background_data_dir: str = "./data/COCO/background_images/", seed: int = 42, num_workers: int = 8,
                 normalize: bool = False, convert_grayscale: bool = False, split_ratio: float = 0.8, resize_dim=(32, 32), resize: bool = True, background_only: bool = False, rand_bg: bool = False, rand_normal_bg: bool = False, bg_aug_max: float = 0.5):

        super().__init__()

        self.batch_size = batch_size
        self.foreground_data_dir = foreground_data_dir
        self.background_data_dir = background_data_dir
        self.normalize = normalize
        self.convert_grayscale = convert_grayscale
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.seed = seed
        self.background_only = background_only
        self.rand_bg = rand_bg
        self.rand_normal_bg = rand_normal_bg
        self.bg_aug_max = bg_aug_max

        transform = []
        if convert_grayscale:
            transform.append(transforms.Grayscale(num_output_channels=1))
        if resize:
            transform.append(transforms.Resize(
                resize_dim, interpolation=Image.BICUBIC))

        transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

    def prepare_data(self):
        self.foreground_trainvaltest = datasets.ImageFolder(
            self.foreground_data_dir, transform=self.transform)
        self.background_trainvaltest = datasets.ImageFolder(
            self.background_data_dir, transform=self.transform)

    def setup(self):
        train_data_len = floor(
            (len(self.foreground_trainvaltest) * self.split_ratio))
        val_data_len = floor(len(self.foreground_trainvaltest)
                             * ((1 - self.split_ratio) * 0.5))
        test_data_len = floor(len(self.foreground_trainvaltest)
                              * ((1 - self.split_ratio) * 0.5))

        redundant = len(self.foreground_trainvaltest) - \
            train_data_len - val_data_len - test_data_len
        train_data_len += redundant

        # train/val/test split. Seed value has been set to same value in order to keep the order btw foreground and background
        self.foreground_train, self.foreground_val, self.foreground_test = random_split(self.foreground_trainvaltest, [
                                                                                        train_data_len, val_data_len, test_data_len], generator=torch.Generator().manual_seed(self.seed))

        self.background_train, self.background_val, self.background_test = random_split(self.background_trainvaltest, [
                                                                                        train_data_len, val_data_len, test_data_len], generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        if self.background_only:
            dataset = self.background_train
        else:
            dataset = ConcatDataset(
                self.background_train,
                self.foreground_train,
                rand_bg=self.rand_bg,
                rand_normal_bg=self.rand_normal_bg,
                bg_aug_max=self.bg_aug_max
            )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        if self.background_only:
            dataset = self.background_val
        else:
            dataset = ConcatDataset(
                self.background_val,
                self.foreground_val,
                rand_bg=self.rand_bg,
                rand_normal_bg=self.rand_normal_bg,
                bg_aug_max=self.bg_aug_max
            )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        if self.background_only:
            dataset = self.background_test
        else:
            dataset = ConcatDataset(
                self.background_test,
                self.foreground_test,
                rand_bg=self.rand_bg,
                rand_normal_bg=self.rand_normal_bg,
                bg_aug_max=self.bg_aug_max
            )
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)
