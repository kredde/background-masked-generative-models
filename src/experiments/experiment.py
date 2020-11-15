from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

class Experiment():
    def __init__(self, model, dataset, experiment_name, callbacks=[], max_epochs=50, background_subtraction=False):
        self.model = model(background_subtraction=background_subtraction).cuda()
        self.dataset = dataset()
        self.background_subtraction = background_subtraction
        self.dataset.prepare_data()
        self.dataset.setup()
        
        self.experiment_name = experiment_name
        self.logger = TensorBoardLogger("lightning_logs", name=experiment_name)
        self.trainer = Trainer(max_epochs=max_epochs, gpus=1, logger=self.logger, callbacks=callbacks, auto_lr_find=True)
    
    def train(self):
        self.trainer.fit(self.model, self.dataset)
    
    def save(self):
        self.trainer.save_checkpoint("models/" + self.experiment_name + ".ckpt")
        
    def load(self):
        self.model = self.model.load_from_checkpoint(checkpoint_path="models/"+ self.experiment_name +".ckpt", background_subtraction=self.background_subtraction).cuda()
        
