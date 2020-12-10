"""
COCOPixelCNN
"""
from src.models.pixelcnn import PixelCNN
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import Adam, RMSprop
from torch.nn import functional as F
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

def maskAConv(c_in=1, c_out=128, k_size=7, stride=1, pad=3):
    """2D Masked Convolution (type A)"""
    return nn.Sequential(
        MaskedConv2d('A', c_in, c_out, k_size, stride, pad, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(True))

class MaskBConvBlock(nn.Module):
    def __init__(self, h=128, k_size=7, stride=1, pad=3):
        """1x1 Conv + 2D Masked Convolution (type B) + 1x1 Conv"""
        super(MaskBConvBlock, self).__init__()

        self.net = nn.Sequential(
            MaskedConv2d('B', h, h, k_size, stride, pad, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x) + x  #Try residual connection

class COCOPixelCNN(PixelCNN):
    def __init__(self, *args, **kwargs):
        super(COCOPixelCNN, self).__init__(*args, **kwargs)

        self.MaskAConv = maskAConv()

        MaskBConv = []
        for i in range(15):
            MaskBConv.append(MaskBConvBlock())
            
        self.MaskBConv = nn.Sequential(*MaskBConv)

        self.out = nn.Sequential(
            nn.Conv2d(128, 256, 1)
        )

        # self.blocks = nn.Sequential(
        #     MaskedConv2d('A', self.in_channels,  128, 7, 1, 3, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
        #         128), nn.ReLU(True),
        #     nn.Conv2d(128, self.in_channel * 256, 1))

    
    def forward(self, x):
        x = self.MaskAConv(x)

        x = self.MaskBConv(x)

        return self.out(x)

    def cross_entropy_loss(self, logits, targets):

        return F.cross_entropy(logits, targets)
        
    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch[0]
        x_mask, _ = train_batch[1]

        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        if self.background_subtraction:  # Set all likelihood values of background pixels to zero
            
            logits = self.subtract_background_likelihood(logits, x_mask.cuda())

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
            
            logits = self.subtract_background_likelihood(logits, x_mask.cuda())

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
            logits = self.subtract_background_likelihood(logits, x_mask.cuda())

        loss = self.cross_entropy_loss(logits, target)

        self.log('test_loss', loss)
        return {'test_loss': loss}
    
    def subtract_background_likelihood(self, logits, target_mask):
        l = logits.clone()
        
        mask = target_mask.clone()
        mask[mask > 0.0] = 1.0 + self.foreground_addition_value
        mask[mask == 0.0] = self.background_subtraction_value
        mask = mask.repeat(1, 256, 1, 1)
        
        l = l * mask
        
        return l
