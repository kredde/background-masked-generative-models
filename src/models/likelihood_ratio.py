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


class LikelihoodRatio(LightningModule):
    """
        Evaluates a model and a background model using likelihood ratios
    """

    def __init__(self, model=None, model_back=None, back_weight=1, img_index=False, *args, **kwargs):
        super(LikelihoodRatio, self).__init__(*args, **kwargs)

        self.model = model
        self.model_back = model_back
        self.back_weight = back_weight
        self.img_index = img_index

    def forward(self, x):
        full = self.model(x)
        back = self.model_back(x)

        # compute ratio of log likelihoods
        return full - (self.back_weight * back)

    def _step(self, batch, _):
        x, y = batch

        if self.img_index:
            x, _ = x

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

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
