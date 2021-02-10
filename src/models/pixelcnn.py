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

    def __init__(self, learning_rate: int = 1e-3, background_subtraction: bool = False, background_subtraction_value: float = 0.0,
                 kernel_size: int = 7, padding: int = 3, in_channels: int = 1, reg: float = 0.0,
                 *args, **kwargs):
        super(PixelCNN, self).__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.background_subtraction = background_subtraction
        self.background_subtraction_value = background_subtraction_value
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.reg = reg

        self.blocks = nn.Sequential(
            MaskedConv2d('A', in_channels,  64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            MaskedConv2d('B', 64, 64, kernel_size, 1, padding, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(True),
            nn.Conv2d(64, 256, 1))

    def forward(self, x):
        return self.blocks(x)

    def _step(self, batch, _):
        x, y = batch

        input = Variable(x.cuda())

        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        # self.foreground_addition_value
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
        loss = self._step(val_batch, batch_idx)

        self.log('test_loss', loss)

        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate, weight_decay=self.reg)
        lr_scheduler = {'scheduler': ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]

    def subtract_background_likelihood(self, logits, target):
        l = logits.clone()
        logit_shape = list(l.shape)
        logit_shape[1] = 1  # assign number of channels to 1

        mask = torch.reshape(torch.clone(target), tuple(logit_shape))
        mask = mask.type(torch.FloatTensor)
        mask[mask > 0] = 1
        mask[mask == 0] = self.background_subtraction_value
        mask = mask.repeat(1, 256, 1, 1)

        l = l * mask.cuda()
        return l, target
