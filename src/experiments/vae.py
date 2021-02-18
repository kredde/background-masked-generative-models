import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

import sys, os
sys.path.append(os.getcwd())

from src.data.bg_aug_mnist import BgAugMNISTDataModule
from src.models.vae.basic_vae_variance import BasicVAEVariance
from src.experiments.experiment import Experiment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(42)


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0000,
    patience=5,
    verbose=True,
)

model_params = {
    'bg_aug_train': True
}
dataset_params = {
    'single_image': True
}

model = BasicVAEVariance
dataset = BgAugMNISTDataModule

exp = Experiment('BasicVAEVariance_AugMNIST_Full_Image', model=model, dataset=dataset, model_params=model_params,
                 dataset_params=dataset_params, callbacks=[early_stop_callback], device=device, max_epochs=25)
exp.setup_new()

exp.train()

exp.save()
