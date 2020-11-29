from pytorch_lightning.core.lightning import LightningModule
from torch import nn, stack
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
import torch
from src.models.vae.utils import kl_loss_function, sample
from src.models.pixelcnn import PixelCNN


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        scale = 2

        layers = [
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((scale, scale)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((scale, scale)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # maxpool 3x3 to get down to 2x2 (needed for 28x28 image upsampling)
            nn.MaxPool2d((3, 3)),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        upmode = 'bilinear'
        scale = 2

        layers = [
            nn.Upsample(scale_factor=scale, mode=upmode),
            nn.ConvTranspose1d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # we need to scale by 1.75 here in order to get 28x28 image size
            # which is not a power of 2
            nn.Upsample(scale_factor=1.75, mode=upmode),
            nn.ConvTranspose1d(32, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=scale, mode=upmode),
            nn.ConvTranspose1d(16, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=scale, mode=upmode),
            nn.ConvTranspose1d(16, 1, kernel_size=(3, 3), padding=1),
            torch.nn.Sigmoid()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, dec_out_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(out_dim=dec_out_dim)

    def forward(self, x):
        x = self.encoder(x)
        # here is where we would reparametriza in a vae
        x = self.decoder(x)
        return x


class Auto(LightningModule):

    def __init__(self, learning_rate: int = 1e-3, dec_out_dim: int = 1, bg_sub: bool = False, bg_sub_val: float = 0.0, pixel_recon: bool = False, * args, **kwargs):
        super(Auto, self).__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.bg_sub = bg_sub
        self.bg_sub_val = bg_sub_val
        self.pixel_recon = pixel_recon

        self.auto = Autoencoder(dec_out_dim=dec_out_dim)
        if self.pixel_recon:
            self.pixelcnn = PixelCNN(in_channels=dec_out_dim)

    def forward(self, x):
        x = self.auto(x)

        if self.pixel_recon:
            return self.pixelcnn(x)
        return x

    def _step(self, batch, idx):
        x, y = batch

        x = Variable(x.cuda())

        logits = self.forward(x)

        # subtract bg likelihoods
        if self.bg_sub:
            mask = x.clone()
            mask[mask > 0.0] = 1
            mask[mask == 0.0] = self.bg_sub_val

            # if we use pixelcnn in recon, we need 256 channels
            if self.pixel_recon:
                mask = mask.repeat(1, 256, 1, 1)
            logits = logits * mask

        # bce for pixel recon loss and mse for plain autoencoder
        if self.pixel_recon:
            target = Variable((x.data[:, 0] * 255).long())
            target = target.squeeze(1).long()
            loss = F.cross_entropy(logits, target)
        else:
            loss = F.mse_loss(logits, x)

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

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler': ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
