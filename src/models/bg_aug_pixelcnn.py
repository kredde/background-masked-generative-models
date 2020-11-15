"""
BgAugPixelCNN
"""
from src.models.pixelcnn import PixelCNN
from torch import nn
from torch.autograd import Variable
import torch
import numpy as np
from src.models.pixelcnn import PixelCNN


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


class BgAugPixelCNN(PixelCNN):
    def training_step(self, train_batch, batch_idx):
        x, x1, x2, y = train_batch

        input = Variable(x1.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)
        loss = self.cross_entropy_loss(logits, target)

        input2 = Variable(x2.cuda())
        target2 = Variable((x.data[:, 0] * 255).long())
        logits2 = self.forward(input2)
        loss2 = self.cross_entropy_loss(logits2, target2)

        loss = ((loss + loss2) / 2) + torch.abs(loss - loss2)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, x1, x2, y = val_batch
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())

        logits = self.forward(input)
        loss = self.cross_entropy_loss(logits, target)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, val_batch, batch_idx):
        x = val_batch[0]
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())

        logits = self.forward(input)
        loss = self.cross_entropy_loss(logits, target)

        self.log('test_loss', loss)
        return {'test_loss': loss}
