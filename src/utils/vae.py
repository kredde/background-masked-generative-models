"""
    Utility functions for VAE
"""
from sklearn.manifold import TSNE
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number
import math
from scipy.stats import multivariate_normal, norm

def generate_images(model, test_data, input=28, random_bg=False, bg_pixel_val=0.5):
    """
        Reconstruct mean and variance using VAE-Variance
    """
    columns = 3
    rows = 1
    fig = plt.figure(figsize=(10, 15))
    i = 1
    for img in iter(test_data):
        if i <= (columns * rows) * 3:
           
            img = img[0][0] if not random_bg else constant_gray_bg(img[0][0], bg_pixel_val)
            mask = torch.flatten(img.clone(), 1)
            
            rec_mu, rec_var = model(img.cuda())

            std = rec_var.sqrt()
            p = torch.distributions.Normal(rec_mu, std)
           
            fig.add_subplot(rows * 3, columns * 3, i)
            plt.imshow(rec_mu.view(input,input).detach().cpu().numpy(), cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.title("mean")
            
            fig.add_subplot(rows * 3, columns * 3, i+1)
            plt.imshow(rec_var.view(input,input).detach().cpu().numpy(), cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.title("var")

            fig.add_subplot(rows * 3, columns * 3, i+2)
            plt.imshow(img[0, :, :].detach().cpu().numpy(), cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.title("original")
            
        i += 3
    plt.show()

def normalize(data, min_range=0.0, max_range=1.0):
    """
        Basic normalization function for input images
    """
    min_pixel = torch.min(data)
    max_pixel = torch.max(data)

    min_range = min_range
    max_range = max_range

    scaled_data = ((data - min_pixel) / (max_pixel - min_pixel) * (max_range - min_range)) + min_range

    return scaled_data

def constant_gray_bg(img, bg_pixel_val=0.5):
    """
        Convert background pixels to gray
    """
    min_pixel_val = torch.min(img)
    img[img == min_pixel_val] = bg_pixel_val
    return img

def randomize_background(img, min_range=0.0, max_range=0.7):
    """
        Change every background pixel value uniformly
    """
    min_pixel_val = torch.min(img)
    img[img == min_pixel_val] = (min_range - max_range) * torch.rand(img[img == min_pixel_val].shape, ) + max_range
    return img


    

