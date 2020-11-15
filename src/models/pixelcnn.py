"""
PixelCNN
"""
from pytorch_lightning.core.lightning import LightningModule
from torch import nn, stack
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN(LightningModule):

    def __init__(self, lr_rate: int = 1e-3, background_subtraction: bool = False):
        super().__init__()

        self.lr_rate = lr_rate
        self.background_subtraction = background_subtraction
        self.blocks = nn.Sequential(
            MaskedConv2d('A', 1,  64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            nn.Conv2d(64, 256, 1))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.blocks(x)

    def cross_entropy_loss(self, logits, targets):
        targets = targets.squeeze(1).long()

        return F.cross_entropy(logits, targets)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)
        
        if self.background_subtraction: # Set all likelihood values of background pixels to zero
            logits, target = self.subtract_background_likelihood(logits, target)
        
        loss = self.cross_entropy_loss(logits, target)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)
        
        if self.background_subtraction: # Set all likelihood values of background pixels to zero
            logits, target = self.subtract_background_likelihood(logits, target)
            
        loss = self.cross_entropy_loss(logits, target)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)
        
        if self.background_subtraction: # Set all likelihood values of background pixels to zero
            logits, target = self.subtract_background_likelihood(logits, target)
        
        loss = self.cross_entropy_loss(logits, target)

        self.log('test_loss', loss)
        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

        return {'avg_val_loss': avg_loss}

    def test_epoch_end(self, outputs):
        avg_loss = stack([x['test_loss'] for x in outputs]).mean()

        self.log('avg_test_loss', avg_loss)
        return {'avg_test_loss': avg_loss, 'loss': [x['test_loss'] for x in outputs]}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = {'scheduler': ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
    
    def subtract_background_likelihood(self, logits, target):
        l = logits.clone()
        logit_shape = list(l.shape)
        logit_shape[1] = 1 # assign number of channels to 1
        
        mask = torch.reshape(torch.clone(target), tuple(logit_shape))
        mask[mask > 0] = 1
        mask = mask.repeat(1, 256, 1, 1) # this should not be static
        
        l = l * mask
        return l, target
