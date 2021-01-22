import numpy as np
import torch

class Mutate(object):
    """Mutates image pixels using a given probability

    args:
        rate: In [0, 1]. The probability for corrupting an image pixel.
    """

    def __init__(self, rate, shape=(1,28,28)):
        self.rate = rate
        self.shape = shape

    def __call__(self, sample):
        """
        args:
            param sample: Image tensor.
        returns: The perturbed image.
        """
        mask = torch.Tensor(np.random.choice([0, 1], sample.shape, p=[1 - self.rate, self.rate]))

        sample = sample * (mask < 1).int() + (torch.rand(sample.shape) * mask)
        return sample.float()



        