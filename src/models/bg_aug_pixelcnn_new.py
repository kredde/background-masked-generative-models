"""
BgAugPixelCNN New
"""
from torch import nn
from torch.autograd import Variable
import torch
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F

from src.models.pixelcnn import PixelCNN
from src.utils.pixelcnn import randomize_background, randomize_background_normal

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

def maskAConv(c_in=1, c_out=64, k_size=7, stride=1, pad=3):
    """2D Masked Convolution (type A)"""
    return nn.Sequential(
        MaskedConv2d('A', c_in, c_out, k_size, stride, pad, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(True))


class MaskBConvBlock(nn.Module):
    def __init__(self, h=64, k_size=7, stride=1, pad=3, residual_connection=False):
        """1x1 Conv + 2D Masked Convolution (type B) + 1x1 Conv"""
        super(MaskBConvBlock, self).__init__()

        self.residual_connection = residual_connection

        self.net = nn.Sequential(
            MaskedConv2d('B', h, h, k_size, stride, pad, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Try residual connection
        return self.net(x) + x if self.residual_connection else self.net(x)


class BgAugPixelCNNNew(LightningModule):
    def __init__(self, learning_rate: int = 1e-3, background_subtraction: bool = False, background_subtraction_value: float = 0.0, foreground_addition_value: float = 0.0, 
                 kernel_size: int = 7, padding: int = 3, in_channels: int = 1, bg_aug: bool = True, bg_aug_max: float = 1.0, residual_connection: bool = False, 
                 *args, **kwargs):
        super(BgAugPixelCNNNew, self).__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.background_subtraction = background_subtraction
        self.background_subtraction_value = background_subtraction_value
        self.foreground_addition_value = foreground_addition_value
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.bg_aug_max = bg_aug_max
        self.bg_aug = bg_aug
        
        maskA_block = maskAConv(c_in=self.in_channels, pad=self.padding, k_size=self.kernel_size)
        maskB_blocks = []
        for i in range(15):
            maskB_blocks.append(MaskBConvBlock(k_size=self.kernel_size, pad=self.padding, residual_connection=residual_connection))
            
        self.blocks = nn.Sequential(maskA_block, *maskB_blocks, nn.Conv2d(64, 256, 1))
    
    def forward(self, x):
        return self.blocks(x)
        
    def _step(self, batch, batch_idx):
        x, y = batch
        
        if self.bg_aug:
            x1 = torch.clone(x)
            x2 = torch.clone(x)

            x1 = randomize_background(x1, norm=self.bg_aug_max)
            x2 = randomize_background(x2, norm=self.bg_aug_max)
            target = randomize_background_normal(x, mean=0.05, std=0.02)

            input = Variable(x1.cuda())
            target = Variable((target.data[:, 0] * 255).long())
            logits = self.forward(input)
            loss = self.cross_entropy_loss(logits, target)

            input2 = Variable(x2.cuda())
            target2 = Variable((x2.data[:, 0] * 255).long())
            logits2 = self.forward(input2)
            loss2 = self.cross_entropy_loss(logits2, target2)

            loss = ((loss + loss2) / 2) + torch.square(loss - loss2)
        else:
            x = batch[0]
            input = Variable(x.cuda())
            target = Variable((x.data[:, 0] * 255).long())
            logits = self.forward(input)

            if self.background_subtraction:
                logits, target = self.subtract_background_likelihood(
                    logits, target)

            loss = self.cross_entropy_loss(logits, target)

        return loss
    
    def cross_entropy_loss(self, logits, targets):
        targets = targets.squeeze(1).long()

        return F.cross_entropy(logits, targets)

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx)

        self.log('val_loss', loss)
        return loss


    def test_step(self, val_batch, batch_idx):
        x = val_batch[0]
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        if self.background_subtraction:
            logits, target = self.subtract_background_likelihood(
                logits, target)

        loss = self.cross_entropy_loss(logits, target)

        self.log('test_loss', loss)
        return {'test_loss': loss}
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler': ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]

    def subtract_background_likelihood(self, logits, target):
        l = logits.clone()
        logit_shape = list(l.shape)
        logit_shape[1] = 1  # assign number of channels to 1

        mask = torch.reshape(torch.clone(target), tuple(logit_shape))
        mask = mask.type(torch.FloatTensor)
        mask[mask > 0] = 1 + self.foreground_addition_value
        mask[mask == 0] = self.background_subtraction_value
        mask = mask.repeat(1, 256, 1, 1)  # this should not be static

        l = l * mask.cuda()
        return l, target