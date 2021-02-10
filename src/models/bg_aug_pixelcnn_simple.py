
from src.models.pixelcnn import PixelCNN
from torch.autograd import Variable
import torch
from src.models.pixelcnn import PixelCNN
from src.utils.pixelcnn import randomize_background


class BgAugPixelCNN(PixelCNN):
    """
        Extends the pixel cnn model with the pair learning approach
    """

    def __init__(self, bg_aug_max: float = 1.0, *args, **kwargs):
        super(BgAugPixelCNN, self).__init__(*args, **kwargs)

        self.bg_aug_max = bg_aug_max

    def training_step(self, train_batch, _):
        x, _ = train_batch

        x1 = torch.clone(x)
        x2 = torch.clone(x)

        x1 = randomize_background(x1, norm=self.bg_aug_max)
        x2 = randomize_background(x2, norm=self.bg_aug_max)

        input = Variable(x1.cuda())
        target = Variable((x1.data[:, 0] * 255).long())
        logits = self.forward(input)
        loss = self.cross_entropy_loss(logits, target)

        input2 = Variable(x2.cuda())
        target2 = Variable((x2.data[:, 0] * 255).long())
        logits2 = self.forward(input2)
        loss2 = self.cross_entropy_loss(logits2, target2)

        loss = ((loss + loss2) / 2) + torch.square(loss - loss2)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, val_batch, _):
        x, y = val_batch

        x1 = torch.clone(x)
        x2 = torch.clone(x)

        x1 = randomize_background(x1, norm=self.bg_aug_max)
        x2 = randomize_background(x2, norm=self.bg_aug_max)

        input = Variable(x1.cuda())
        target = Variable((x1.data[:, 0] * 255).long())
        logits = self.forward(input)
        loss = self.cross_entropy_loss(logits, target)

        input2 = Variable(x2.cuda())
        target2 = Variable((x2.data[:, 0] * 255).long())
        logits2 = self.forward(input2)

        loss2 = self.cross_entropy_loss(logits2, target2)
        loss = ((loss + loss2) / 2) + torch.square(loss - loss2)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, val_batch, _):
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
