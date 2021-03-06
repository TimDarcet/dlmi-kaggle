import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl


# Some code reused from https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
class TransferResNet(pl.LightningModule):
    def __init__(self,
                 resnet_version=18,
                 optimizer='adam',
                 lr=1e-3):
        super().__init__()
        self.__dict__.update(locals())

        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}

        self.optimizer = optimizers[optimizer]

        self.resnet = resnets[resnet_version](pretrained=True)
        linear_size = list(self.resnet.children())[-1].in_features
        self.resnet.fc = nn.Linear(linear_size, 1)

        for child in list(self.resnet.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = False
        self.resnet.eval()

        self.alpha = 4.2  # Comes from dataset imbalance, factor to rebalance the loss

    def forward(self, im):
        return self.resnet(im).squeeze()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        image, label, _ = batch
        score = self(image)
        weights = 1 / (label.float() * (self.alpha - 1) + 1)
        loss = nn.BCEWithLogitsLoss(weights)(score, label.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label, _ = batch
        score = self(image)
        weights = 1 / (label.float() * (self.alpha - 1) + 1)
        loss = nn.BCEWithLogitsLoss(weights)(score, label.float())
        
        pred = (score > 0).int()
        acc = (pred == label).sum() / pred.shape[0]
        
        # Balanced accuracy
        ones = label == 1
        if pred[ones].shape[0] > 0:
            acc1 = (pred[ones] == label[ones]).sum() / pred[ones].shape[0]
        else:
            acc1 = 0.5
        zeros = label == 0
        if pred[zeros].shape[0] > 0:
            acc0 = (pred[zeros] == label[zeros]).sum() / pred[zeros].shape[0]
        else:
            acc0 = 0.5
        accbal = (acc0 + acc1 ) / 2
        
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_acc1", acc1, prog_bar=True)
        self.log("val_acc0", acc0, prog_bar=True)
        self.log("val_accbal", accbal, prog_bar=True)
        
        return loss


class TransferFeaturizer(pl.LightningModule):
    def __init__(self,
                 resnet_version=18,
                 optimizer='adam',
                 lr=1e-3):
        super().__init__()
        self.__dict__.update(locals())

        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}

        self.optimizer = optimizers[optimizer]

        self.resnet = resnets[resnet_version](pretrained=True)
        linear_size = list(self.resnet.children())[-1].in_features
        self.resnet.fc = nn.Linear(linear_size, 64)
        self.classif = nn.Linear(64, 1)

        for child in list(self.resnet.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = False
        self.resnet.eval()

        self.alpha = 4.2  # Comes from dataset imbalance, factor to rebalance the loss

    def forward(self, im):
        h = torch.relu(self.resnet(im))
        return self.classif(h).squeeze()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        image, label, _ = batch
        score = self(image)
        weights = 1 / (label.float() * (self.alpha - 1) + 1)
        loss = nn.BCEWithLogitsLoss(weights)(score, label.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label, _ = batch
        score = self(image)
        weights = 1 / (label.float() * (self.alpha - 1) + 1)
        loss = nn.BCEWithLogitsLoss(weights)(score, label.float())
        
        pred = (score > 0).int()
        acc = (pred == label).sum() / pred.shape[0]
        
        # Balanced accuracy
        ones = label == 1
        if pred[ones].shape[0] > 0:
            acc1 = (pred[ones] == label[ones]).sum() / pred[ones].shape[0]
        else:
            acc1 = 0.5
        zeros = label == 0
        if pred[zeros].shape[0] > 0:
            acc0 = (pred[zeros] == label[zeros]).sum() / pred[zeros].shape[0]
        else:
            acc0 = 0.5
        accbal = (acc0 + acc1 ) / 2
        
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_acc1", acc1, prog_bar=True)
        self.log("val_acc0", acc0, prog_bar=True)
        self.log("val_accbal", accbal, prog_bar=True)
        
        return loss
