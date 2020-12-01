import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_images(model, channels=1):
    sample = torch.Tensor(144, channels, 28, 28).cuda()

    model.cuda()
    sample.fill_(0)
    model.train(False)
    for i in range(28):
        for j in range(28):
            out = model(Variable(sample))
            probs = F.softmax(out[:, :, i, j]).data
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

    fig = plt.figure(figsize=(8, 8))

    columns = 12
    rows = 12
    for i in range(1, columns*rows + 1):
        if i < 144:
            fig.add_subplot(rows, columns, i)
            plt.imshow(sample.detach().cpu().numpy()[i][0], cmap='gray')
            plt.xticks([])
            plt.yticks([])
    plt.show()


def randomize_background(img, norm=0.5):
    img[img == 0] = np.random.random_sample() * norm
    return img


def likelihood(img_data, model):
    img = img_data[0].cuda()
    if hasattr(model, 'position_encode') and model.position_encode:
        img = model.positional_encoding(img)
    img = img.cuda()
    model.eval()
    res = model(img)
    like = torch.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            probs = F.softmax(res[0, :, i, j], dim=0)
            prob = (img[0, :, i, j] * 255.).int().cpu().numpy()[0]
            like[i][j] = probs[prob]

    return like


def draw_likelihood_plot(data, model):
    columns = 4
    rows = 4
    fig = plt.figure(figsize=(8, 8))
    i = 1
    for img in iter(data):
        if i <= 16:
            fig.add_subplot(rows, columns, i)
            like = likelihood(img, model)
            sns.heatmap(like.detach().cpu().numpy(), cmap="gray", vmax=.1)
            plt.xticks([])
            plt.yticks([])
        i += 1
    plt.show()
