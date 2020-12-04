"""
COCOPixelCNN
"""
from src.models.pixelcnn import PixelCNN
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
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


def maskAConv(c_in=1, c_out=128, k_size=7, stride=1, pad=3):
    """2D Masked Convolution (type A)"""
    return nn.Sequential(
        MaskedConv2d('A', c_in, c_out, k_size, stride, pad, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(True))


class MaskBConvBlock(nn.Module):
    def __init__(self, h=128, k_size=7, stride=1, pad=3, residual_connection=False):
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


class COCOPixelCNN(PixelCNN):
    def __init__(self, learning_rate: int = 1e-3, background_subtraction: bool = False, background_subtraction_value: float = 0.0, kernel_size: int = 7, padding: int = 3, in_channels: int = 1, concat_dataset: bool = True, bg_aug: bool = False, residual_connection: bool = False, *args, **kwargs):
        super(COCOPixelCNN, self).__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.background_subtraction = background_subtraction
        self.background_subtraction_value = background_subtraction_value
        self.in_channels = in_channels
        self.concat_dataset = concat_dataset
        self.bg_aug = bg_aug

        self.MaskAConv = maskAConv()

        MaskBConv = []
        for i in range(15):
            MaskBConv.append(MaskBConvBlock(
                residual_connection=residual_connection))

        self.MaskBConv = nn.Sequential(*MaskBConv)

        self.out = nn.Sequential(
            nn.Conv2d(128, 256, 1)
        )

    def forward(self, x):
        x = self.MaskAConv(x)

        x = self.MaskBConv(x)

        return self.out(x)

    def cross_entropy_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)

    def _step(self, batch, batch_idx):
        if self.concat_dataset:
            background_batch, foreground_batch = batch
        else:
            background_batch = batch
        x, _ = background_batch

        if self.bg_aug:
            x_foreground, _ = foreground_batch
            target = Variable((x.data[:, 0] * 255).long())

            x1 = Variable(x.cuda())
            logits = self.forward(x1)
            loss = self.cross_entropy_loss(logits, target)

            x2 = Variable(x_foreground.cuda())
            logits2 = self.forward(x2)
            loss2 = self.cross_entropy_loss(logits2, target)

            loss = ((loss + loss2) / 2) + torch.square(loss - loss2)

        else:
            input = Variable(x.cuda())
            target = Variable((x.data[:, 0] * 255).long())
            logits = self.forward(input)

            if self.background_subtraction:
                mask, _ = foreground_batch
                logits = self.subtract_background_likelihood(
                    logits, mask.cuda())

            loss = self.cross_entropy_loss(logits, target)

        return loss

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

    def subtract_background_likelihood(self, logits, target_mask):
        l = logits.clone()

        mask = target_mask.clone()
        mask[mask > 0.0] = 1.0 + self.foreground_addition_value
        mask[mask == 0.0] = self.background_subtraction_value
        mask = mask.repeat(1, 256, 1, 1)

        l = l * mask

        return l

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler': ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
