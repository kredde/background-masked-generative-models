import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_images(model, channels=1, img_dim=(28,28)):
    sample = torch.Tensor(64, channels, *img_dim).cuda()

    model.cuda()
    sample.fill_(0)
    model.train(False)
    for i in range(img_dim[0]):
        for j in range(img_dim[1]):
            out = model(Variable(sample))
            probs = F.softmax(out[:, :, i, j]).data
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

    fig = plt.figure(figsize=(8, 8))

    columns = 8
    rows = 8
    for i in range(1, columns*rows + 1):
        if i < 64:
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
    b, c, w, h = res.shape
    like = torch.zeros((w, h))
    for i in range(w):
        for j in range(h):
            probs = F.softmax(res[0, :, i, j], dim=0)
            prob = (img[0, :, i, j] * 255.).int().cpu().numpy()[0]
            like[i][j] = probs[prob]

    return like


def draw_likelihood_plot(data, model, cmap="gray", vmax=.1, img_index=None, dim=(4,4)):
    columns, rows = dim
    fig = plt.figure(figsize=(16, 16))
    i = 1
    for img in iter(data):
        if i <= (columns * rows) * 2:
            fig.add_subplot(rows * 2, columns * 2, i)
            like = likelihood(img if img_index == None else img[img_index], model)
            sns.heatmap(like.detach().cpu().numpy(), cmap=cmap, vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            
            fig.add_subplot(rows * 2, columns* 2, i + 1)
            plt.imshow((img if img_index == None else img[img_index])[0][0][0], cmap="gray")
            plt.xticks([])
            plt.yticks([])
        i += 2
    plt.show()

