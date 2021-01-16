"""
PixelCNN
"""
from pytorch_lightning.core.lightning import LightningModule
from torch import nn, stack
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
from src.utils.pixelcnn import positionalencoding2d
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

    def __init__(self, learning_rate: int = 1e-3, background_subtraction: bool = False, background_subtraction_value: float = 0.0, foreground_addition_value: float = 0.0, kernel_size: int = 7, padding: int = 3, in_channels: int = 1, position_encode: bool = False, mse: bool = False, fg_mse: bool = False, reg: float = 0.0, *args, **kwargs):
        super(PixelCNN, self).__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.background_subtraction = background_subtraction
        self.background_subtraction_value = background_subtraction_value
        self.foreground_addition_value = foreground_addition_value
        self.position_encode = position_encode
        self.in_channels = in_channels
        self.mse = mse
        self.fg_mse = fg_mse
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

    def _step(self, batch, batch_idx):
        x, y = batch

        input = Variable(x.cuda())
        if self.position_encode:
            input = self.positional_encoding(input)
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        if self.background_subtraction:  # Set all likelihood values of background pixels to zero
            logits, target = self.subtract_background_likelihood(
                logits, target)
        loss = self.cross_entropy_loss(logits, target)

        if self.mse:
            mse = self.compute_mse(logits, target, masked=self.fg_mse)
            loss += mse

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
        mask[mask > 0] = 1 + self.foreground_addition_value
        mask[mask == 0] = self.background_subtraction_value
        mask = mask.repeat(1, 256, 1, 1)  # this should not be static

        l = l * mask.cuda()
        return l, target

    def compute_mse(self, logits, target, masked=False):
        values = logits.max(1).indices
        if masked:
            mask = target.clone()
            mask[mask > 0] = 1
            values = values * mask

        values = values.type(torch.FloatTensor) / 255
        target = target.type(torch.FloatTensor) / 255

        return F.mse_loss(values, target)

    def positional_encoding(self, batch):
        pe = positionalencoding2d(4, *batch.shape[2:4])
        pe = pe.repeat(batch.shape[0], 1, 1, 1)
        encoded = torch.cat((batch, pe[:, 0].view(batch.shape), pe[:, 1].view(batch.shape), pe[:, 2].view(
            batch.shape), pe[:, 3].view(batch.shape))).view(batch.shape[0], 5, *batch.shape[2:4])
        return encoded
