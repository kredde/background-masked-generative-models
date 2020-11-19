import pickle
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class Experiment():
    def __init__(self, experiment_name, model=None, dataset=None, callbacks=[], model_params={}, dataset_params={},
                 max_epochs=50, background_subtraction=False):
        """
            usage:
                # setup a new experiment
                exp = Experiment('PixelCNN_MNIST_1', model=PixelCNN,
                                 dataset=MNISTDataModule, callbacks=early_stopping)
                exp.setup_new()
                exp.train()
                exp.save()

                # load existing experiment
                exp = Experiment('PixelCNN_MNIST_1')
                exp.load()
        """
        self.model_params = model_params
        self.dataset_params = dataset_params
        self.model_class = model
        self.dataset_class = dataset
        self.experiment_name = experiment_name
        self.max_epochs = max_epochs
        self.callbacks = callbacks
        self.background_subtraction = background_subtraction

    def _setup(self):
        """
            sets up the model and dataset with the given params
        """
        self.model = self.model_class(**self.model_params).cuda()
        self.dataset = self.dataset_class(**self.dataset_params)
        self.dataset.prepare_data()
        self.dataset.setup()
        self.trainer = Trainer(max_epochs=self.max_epochs, gpus=1, callbacks=self.callbacks, auto_lr_find=True)

    def setup_new(self):
        """
            sets up a new experiment
        """
        self._setup()

        self.logger = TensorBoardLogger(
            'lightning_logs', name=self.experiment_name)

        self.trainer = Trainer(max_epochs=self.max_epochs, gpus=1,
                               logger=self.logger, callbacks=self.callbacks, auto_lr_find=True)

    def train(self):
        self.trainer.fit(self.model, self.dataset)

    def save(self):
        """
            saves the configuration
        """
        config = {
            'dataset': self.dataset_params,
            'model': self.model_params,
            'max_epochs': self.max_epochs,
            'model_class': self.model_class,
            'dataset_class': self.dataset_class,
            'callbacks': self.callbacks
        }
        path = 'models/' + self.experiment_name
        Path('models/' + self.experiment_name).mkdir(parents=True, exist_ok=True)
        with open(path + '/config.p', 'wb') as pickle_file:
            pickle.dump(config, pickle_file)

        self.trainer.save_checkpoint(
            'models/' + self.experiment_name + '.ckpt')

    def load(self):
        """
            loads a given model
        """
        path = Path('models/' + self.experiment_name)

        if (path / 'config.p').is_file():
            with open(str(path / 'config.p'), 'rb') as pickle_file:
                config = pickle.load(pickle_file)
                self.dataset_params = config['dataset']
                self.model_params = config['model']
                self.max_epochs = config['max_epochs']
                self.model_class = config['model_class'] if 'model_class' in config.keys() else self.model_class
                self.dataset_class = config['dataset_class']
                self.callbacks = config['callbacks']
                self._setup()
        
        filename = self.experiment_name + '.ckpt'
        if (path / filename).is_file():
            model_path = str(path / filename)
        else:
            model_path = str('models/' + self.experiment_name + '.ckpt')
            
        self.model = self.model.load_from_checkpoint(checkpoint_path=model_path, **self.model_params).cuda()