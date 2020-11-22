"""
COCOPixelCNN
"""
from src.models.pixelcnn import PixelCNN
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt


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


class COCOPixelCNN(PixelCNN):
    def __init__(self, *args, **kwargs):
        super(COCOPixelCNN, self).__init__(*args, **kwargs)
        
    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch[0]
        x_mask, _ = train_batch[1]
        
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        if self.background_subtraction:  # Set all likelihood values of background pixels to zero
            
            logits = self.subtract_background_likelihood(logits, x, x_mask.cuda())

        loss = self.cross_entropy_loss(logits, target)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch[0]
        x_mask, _ = val_batch[1]
        
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        if self.background_subtraction:  # Set all likelihood values of background pixels to zero
            
            logits = self.subtract_background_likelihood(logits, x, x_mask.cuda())

        loss = self.cross_entropy_loss(logits, target)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        x, _ = test_batch[0]
        x_mask, _ = test_batch[1]
        
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        if self.background_subtraction:  # Set all likelihood values of background pixels to zero
            
            logits = self.subtract_background_likelihood(logits, x, x_mask.cuda())

        loss = self.cross_entropy_loss(logits, target)

        self.log('test_loss', loss)
        return {'test_loss': loss}
    
    def subtract_background_likelihood(self, logits, original_img, target_mask):
        l = logits.clone()
        
        mask = target_mask.clone()
        mask[mask > 0.0] = 1
        mask[mask == 0.0] = self.background_subtraction_value
        mask = mask.repeat(1, 256, 1, 1)
        
        l = l * mask
        
        return l