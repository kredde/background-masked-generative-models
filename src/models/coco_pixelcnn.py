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
    def __init__(self, learning_rate: int = 1e-3, background_subtraction: bool = False, background_subtraction_value: float = 0.0,
                 kernel_size: int = 7, padding: int = 3, in_channels: int = 1, concat_dataset: bool = True, bg_aug: bool = False,
                 random_bg: bool = False, target_random: bool = False, single_loss: bool = False, residual_connection: bool = False,
                 mse_loss: bool = False, target_background: bool = False, target_fg: bool = False, random_normal_bg_target: bool = False, reg: float = 0.0,
                 svhn: bool = False,
                 *args, **kwargs):

        super(COCOPixelCNN, self).__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.background_subtraction = background_subtraction
        self.background_subtraction_value = background_subtraction_value
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.concat_dataset = concat_dataset
        self.bg_aug = bg_aug
        self.random_bg = random_bg
        self.target_random = target_random
        self.single_loss = single_loss
        self.mse_loss = mse_loss
        self.target_fg = target_fg
        self.random_normal_bg_target = random_normal_bg_target
        self.reg = reg
        self.svhn = svhn

        # set up residual connection blocks
        if residual_connection:
            MaskAConv = maskAConv(
                c_in=self.in_channels, pad=self.padding, k_size=self.kernel_size)
            MaskBConv = []
            for i in range(15):
                MaskBConv.append(MaskBConvBlock(k_size=self.kernel_size, pad=self.padding,
                                                residual_connection=residual_connection))
            self.blocks = nn.Sequential(
                MaskAConv, *MaskBConv, nn.Conv2d(128, 256, 1))
        else:
            # set up normal blocks for backwards compatibility of older models
            self.blocks = nn.Sequential(
                MaskedConv2d('A', self.in_channels,  128, 7, 1, 3, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                MaskedConv2d('B', 128, 128, self.kernel_size, 1, self.padding, bias=False), nn.BatchNorm2d(
                    128), nn.ReLU(True),
                nn.Conv2d(128, self.in_channels * 256, 1))

    def forward(self, x):
        return self.blocks(x)

    def cross_entropy_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)

    def _step(self, batch, batch_idx):
        if self.concat_dataset:
            background_batch, foreground_batch = batch
            x_foreground = foreground_batch[0]
        else:
            background_batch = batch
        x = background_batch[0]
        if self.bg_aug:
            if self.random_bg:
                x_foreground, x_foreground_randbg, _ = foreground_batch

                if self.target_random:
                    target = x_foreground_randbg.clone()
                else:
                    if self.random_normal_bg_target:
                        target = randomize_background_normal(
                            x_foreground, mean=0.02, std=0.01)
                    else:
                        target = x_foreground
                x2 = Variable(x_foreground_randbg.cuda())

            else:
                x_foreground, _ = foreground_batch
                target = x_foreground.clone()

                x2 = Variable(x_foreground.cuda())

            target = Variable((target.data[:, 0] * 255).long())

            if self.single_loss:
                logits = self.forward(x2)
                loss = self.cross_entropy_loss(logits, target)
            elif self.mse_loss:
                target_mse = target.unsqueeze(1)
                one_hot = torch.cuda.FloatTensor(target_mse.size(
                    0), 256, target_mse.size(2), target_mse.size(3)).zero_()
                target_mse = one_hot.scatter_(1, target_mse.data, 1)

                x1 = Variable(x.cuda())
                logits = self.forward(x1)
                loss = F.mse_loss(logits, target_mse)

                logits2 = self.forward(x2)
                loss2 = F.mse_loss(logits2, target_mse)

                loss = ((loss + loss2) / 2) + torch.square(loss - loss2)
            else:
                x1 = Variable(x.cuda())
                logits = self.forward(x1)
                loss = self.cross_entropy_loss(logits, target)

                logits2 = self.forward(x2)
                loss2 = self.cross_entropy_loss(logits2, target)

                loss = ((loss + loss2) / 2) + torch.square(loss - loss2)

        else:
            input = Variable(x.cuda())
            if self.target_fg:
                target = Variable((x_foreground.data[:, 0] * 255).long())
            else:
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
        if self.concat_dataset:
            background_batch, foreground_batch = val_batch
        else:
            background_batch = val_batch
        if self.svhn:
            x, _ = val_batch
        else:
            x, _ = background_batch

        input = Variable(x.cuda())
        if self.target_fg:
            target = Variable((x_foreground.data[:, 0] * 255).long())
        else:
            target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

#         if self.background_subtraction:
#             mask = foreground_batch[0]
#             logits = self.subtract_background_likelihood(
#                 logits, mask.cuda())

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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate, weight_decay=self.reg)
        lr_scheduler = {'scheduler': ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
