import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def generate_images(model):
    sample = torch.Tensor(144, 1, 28, 28).cuda()
    
    model.cuda()
    sample.fill_(0)
    model.train(False)
    for i in range(28):
        for j in range(28):
            out = model(Variable(sample))
            probs = F.softmax(out[:, :, i, j]).data
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
    
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))

    columns = 12
    rows = 12
    for i in range(1, columns*rows +1):
        if i < 144:
            fig.add_subplot(rows, columns, i)
            plt.imshow(sample.detach().cpu().numpy()[i][0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
    plt.show()