import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything

from src.models.coco_pixelcnn import COCOPixelCNN
from src.data.coco import COCODataModule
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
    'concat_dataset': True,
    'bg_aug': True,
    'random_bg': True,
    'random_normal_bg_target': True,
    'target_random': True,
    'residual_connection': True,
}
dataset_params = {
    'foreground_data_dir': '/nfs/students/winter-term-2020/project-4/yurtkulus/project-4/data/COCO/foreground_images/person',
    'background_data_dir': '/nfs/students/winter-term-2020/project-4/yurtkulus/project-4/data/COCO/background_images/person',
    'batch_size': 64,
    'convert_grayscale': True,
    'resize': True,
    'background_only': False,
    'rand_normal_bg': True,
    'rand_bg': True,
    'bg_aug_max': 0.2,
    'normalize': True
}

model = COCOPixelCNN
dataset = COCODataModule


exp = Experiment('BgAugPixelCNN_COCO', model=model, dataset=dataset, model_params=model_params,
                 dataset_params=dataset_params, callbacks=[early_stop_callback, lr_monitor], device=device)
exp.setup_new()


# find optimal learning rate
lr_finder = exp.trainer.tuner.lr_find(exp.model, exp.dataset)
lr_finder.results
new_lr = lr_finder.suggestion()
exp.model.lr = new_lr


exp.train()

exp.save()
