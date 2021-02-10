"""
    Utility functions for pixelcnn
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import torch


def generate_images(model, channels=1, img_dim=(28, 28)):
    """
        Generate pixelcnn images
    """
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
    """
        Replace black pixels with a constant random number
    """
    img = img.clone()
    img[img == 0] = np.random.random_sample() * norm
    return img


def randomize_background_normal(foreground, mean=0.4420, std=0.2413):
    """
        Replace black pixels with a random normal distribution
    """
    bg = torch.empty((foreground.shape)).normal_(mean=mean, std=std).cuda()
    bg[bg < 0] = 0.0
    bg[bg > 1] = 1.0

    mask = foreground.clone()
    mask[mask == 0] = 2
    mask[mask <= 1] = 0
    mask[mask == 2] = 1

    bg_masked = bg * mask

    img = bg_masked + foreground

    return img


def likelihood(img_data, model):
    """
        Compute the likelihood of the actual pixels of an image
    """

    img = img_data[0].cuda()

    model.eval()
    # run the image through the model
    res = model(img)
    b, c, w, h = res.shape
    like = torch.zeros((w, h))

    # get the likelihood of the ground truth pixels
    for i in range(w):
        for j in range(h):
            probs = res[0, :, i, j]

            prob = (img[0, :, i, j] * 255.).int().cpu().numpy()[0]
            like[i][j] = probs[prob]

    return like


def draw_likelihood_plot(data, model, cmap="gray", vmax=.1, img_index=None, dim=(4, 4)):
    """
        Plot the likelihood heatmap for a given number of images
    """

    columns, rows = dim
    fig = plt.figure(figsize=(16, 16))
    i = 1
    for img in iter(data):
        if i <= (columns * rows) * 2:
            fig.add_subplot(rows * 2, columns * 2, i)
            like = likelihood(img if img_index ==
                              None else img[img_index], model)
            sns.heatmap(like.detach().cpu().numpy(),
                        cmap=cmap, vmax=vmax, vmin=0)
            plt.xticks([])
            plt.yticks([])

            fig.add_subplot(rows * 2, columns * 2, i + 1)
            plt.imshow((img if img_index == None else img[img_index])[
                       0][0][0], cmap="gray")
            plt.xticks([])
            plt.yticks([])
        i += 2
    plt.show()


def draw_likelihood_plot_ratio(data, model_full, model_back, cmap="gray", vmax=.1, img_index=None, dim=(4, 4)):
    """
        Plot the likelihood heatmap for a likelihood ratio model
    """
    columns, rows = dim
    fig = plt.figure(figsize=(16, 16))
    i = 1
    for img in iter(data):
        if i <= (columns * rows) * 2:
            fig.add_subplot(rows * 2, columns * 2, i)
            like = likelihood_ratio(
                img if img_index == None else img[img_index], model_full, model_back)

            sns.heatmap(like.detach().cpu().numpy(),
                        cmap=cmap, vmax=vmax, vmin=0)
            plt.xticks([])
            plt.yticks([])

            fig.add_subplot(rows * 2, columns * 2, i + 1)
            plt.imshow((img if img_index == None else img[img_index])[
                       0][0][0], cmap="gray")
            plt.xticks([])
            plt.yticks([])
        i += 2
    plt.show()


def likelihood_ratio(img_data, model_full, model_back):
    """
        Compute the likelihood of the actual pixels of an image using likelihood ratios
    """
    img = img_data[0].cuda()

    img = img.cuda()
    model_full.eval()
    res = model_full(img)

    model_back.eval()
    res_back = model_back(img)

    b, c, w, h = res.shape
    like = torch.zeros((w, h))
    for i in range(w):
        for j in range(h):
            probs1 = res[0, :, i, j]
            probs2 = res_back[0, :, i, j]

            probs = probs1 - (1 * probs2)

            prob = (img[0, :, i, j] * 255.).int().cpu().numpy()[0]
            like[i][j] = probs[prob]

    return like


def positionalencoding2d(d_model, height, width):
    """
    Args:
        d_model: dimension of the model
        height: height of the positions
        width: width of the positions

    return:
        d_model*height*width position matrix
    """

    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))

    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.cuda()
