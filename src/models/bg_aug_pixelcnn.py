from torch import nn
from torch.autograd import Variable
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F

from src.utils.pixelcnn import randomize_background, randomize_background_normal
from src.models.blocks.masked_convolution import maskAConv, MaskBConvBlock


class BgAugPixelCNN(LightningModule):
    """
        Model for the pair learning approach
    """

    def __init__(self, learning_rate: int = 1e-3, kernel_size: int = 7, padding: int = 3, in_channels: int = 1, bg_aug: bool = True, bg_aug_max: float = 1.0, residual_connection: bool = False, *args, **kwargs):
        super(BgAugPixelCNN, self).__init__(*args, **kwargs)

        # model params
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # pair learning params
        self.bg_aug_max = bg_aug_max
        self.bg_aug = bg_aug

        maskA_block = maskAConv(c_in=self.in_channels,
                                pad=self.padding, k_size=self.kernel_size)
        maskB_blocks = []
        for _ in range(15):
            maskB_blocks.append(MaskBConvBlock(
                k_size=self.kernel_size, pad=self.padding, residual_connection=residual_connection))

        self.blocks = nn.Sequential(
            maskA_block, *maskB_blocks, nn.Conv2d(64, 256, 1))

    def forward(self, x):
        return self.blocks(x)

    def _step(self, batch, _):
        """
            Exectutes one step of training/testing and returns the loss
        """
        x, _ = batch

        # pair learning
        if self.bg_aug:
            x1 = torch.clone(x)
            x2 = torch.clone(x)

            # prepare inputs
            x1 = randomize_background(x1, norm=self.bg_aug_max)
            x2 = randomize_background(x2, norm=self.bg_aug_max)
            # target is augmented with normal background
            target = randomize_background_normal(x, mean=0.05, std=0.02)

            # first input
            input = Variable(x1.cuda())
            target = Variable((target.data[:, 0] * 255).long())
            logits = self.forward(input)
            loss = self.cross_entropy_loss(logits, target)

            # second input
            input2 = Variable(x2.cuda())
            target2 = Variable((x2.data[:, 0] * 255).long())
            logits2 = self.forward(input2)
            loss2 = self.cross_entropy_loss(logits2, target2)

            # compute total loss
            loss = ((loss + loss2) / 2) + torch.square(loss - loss2)
        else:
            # normal pixelcnn
            x = batch[0]
            input = Variable(x.cuda())
            target = Variable((x.data[:, 0] * 255).long())
            logits = self.forward(input)

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

    def test_step(self, val_batch, _):
        x = val_batch[0]
        input = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        logits = self.forward(input)

        loss = self.cross_entropy_loss(logits, target)

        self.log('test_loss', loss)
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler': ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
