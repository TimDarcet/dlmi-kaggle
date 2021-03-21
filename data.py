from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import pytorch_lightning as pl

import pandas as pd
    

class lymph_dataset(Dataset):
    """Wrapper around an imagefolder dataset to provide with tabular data"""
    def __init__(self, path, csv_path, transforms):
        super().__init__()
        self.path = path
        self.csv_path = csv_path
        self.transforms = transforms
        self.images = torchvision.datasets.ImageFolder(root=self.path, transform=self.transforms)
        self.tabular = pd.read_csv(self.csv_path, index_col=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, lab_idx = self.images[idx]
        patient_id = self.images.classes[lab_idx]
        tab_data = self.tabular.loc[patient_id]
        return image, tab_data.LABEL, int(patient_id[1:])


class lymph_datamodule(pl.LightningDataModule):
    def __init__(self, path, batch_size=32, train_prop=0.8):
        super().__init__()
        self.train_path = join(path, "trainset")
        self.train_csv_path = join(self.train_path, "trainset_true.csv")
        self.test_path = join(path, "testset")
        self.test_csv_path = join(self.test_path, "testset_data.csv")
        self.batch_size = batch_size
        self.train_prop = train_prop

        # TODO data augment
        # TODO normalize?
        # TODO Balance dataset
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor()
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.full_dataset = lymph_dataset(self.train_path, self.train_csv_path, self.train_transforms)
            train_size = int(len(self.full_dataset) * self.train_prop)
            val_size = len(self.full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = lymph_dataset(self.test_path, self.test_csv_path, self.test_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=2)
