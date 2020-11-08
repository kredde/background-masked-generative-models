from src.models.pixelcnn import PixelCNN
from src.data.mnist import MNISTDataModule
from pytorch_lightning import Trainer


mnistdata = MNISTDataModule()
mnistdata.prepare_data()
mnistdata.setup()
model = PixelCNN(input_channels=1)
trainer = Trainer(max_epochs=5)

trainer.fit(model, mnistdata)
