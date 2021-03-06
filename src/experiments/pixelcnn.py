import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything

from src.data.fashionmnist import FashionMNISTDataModule
from src.models.bg_aug_pixelcnn import BgAugPixelCNN
from src.experiments.experiment import Experiment
from pytorch_lightning.callbacks import LearningRateMonitor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(42)


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0000,
    patience=5,
    verbose=True,
)
lr_monitor = LearningRateMonitor(logging_interval='step')

model_params = {
    'bg_aug': True,
    'bg_aug_max': 0.5,
    'residual_connection': True
}
dataset_params = {
}

model = BgAugPixelCNN
dataset = FashionMNISTDataModule


exp = Experiment('BgAugPixelCNN_FASHION', model=model, dataset=dataset, model_params=model_params,
                 dataset_params=dataset_params, callbacks=[early_stop_callback, lr_monitor], device=device)
exp.setup_new()


# find optimal learning rate
lr_finder = exp.trainer.tuner.lr_find(exp.model, exp.dataset)
lr_finder.results
new_lr = lr_finder.suggestion()
exp.model.lr = new_lr


exp.train()

exp.save()
