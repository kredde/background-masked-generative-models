from sklearn.manifold import TSNE
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number
import math
from scipy.stats import multivariate_normal, norm

def generate_images(model, test_data, input=28, random_bg=False):
    columns = 3
    rows = 3
    fig = plt.figure(figsize=(10, 15))
    i = 1
    for img in iter(test_data):
        if i <= (columns * rows) * 3:
           
            img = img[0][0] if not random_bg else constant_gray_bg(img[0][0])
            # img = img[0] # Just for constant images
            mask = torch.flatten(img.clone(), 1)
            
            rec_mu, rec_var = model(img.cuda())

            std = rec_var.sqrt()
            p = torch.distributions.Normal(rec_mu, std)
            recon = p.sample()

            # fig.add_subplot(rows * 4, columns * 4, i)
            # plt.imshow(recon.view(input,input).detach().cpu().numpy(), cmap="gray")
            # plt.xticks([])
            # plt.yticks([])
            # plt.title("recon")
           
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

def reconstructed_probability(img, model, random_bg=False, L=10):
    reconstructed_prob = 0

    if random_bg:
        img = constant_gray_bg(img)

    mu_z, log_var_z = model.encode(img.cuda())

    img = torch.squeeze(torch.flatten(img, 1))
    
    for _ in range(L):
        _, _, z = model.sample_enc(mu_z, log_var_z)
        mu_hat, log_var_hat = model.decode(z)
        # std = var_hat.sqrt() + 1e-5
        var_hat = torch.exp(log_var_hat)
        
        mu_hat = torch.squeeze(mu_hat)
        var_hat = torch.squeeze(var_hat)

        # mu_hat = mu_hat.detach().cpu().numpy()
        # var_hat = var_hat.detach().cpu().numpy()

        dist = torch.distributions.MultivariateNormal(mu_hat, torch.diag(var_hat))
        # prob = multivariate_normal.pdf(img, mu_hat, np.diag(var_hat))
        # reconstructed_prob += prob
        reconstructed_prob += dist.log_prob(img.cuda())
    
    reconstructed_prob /= L
    return reconstructed_prob.item()

def normalize(data, min_range=0.0, max_range=1.0):
    min_pixel = torch.min(data)
    max_pixel = torch.max(data)

    min_range = min_range
    max_range = max_range

    scaled_data = ((data - min_pixel) / (max_pixel - min_pixel) * (max_range - min_range)) + min_range

    return scaled_data

def constant_gray_bg(img):
    min_pixel_val = torch.min(img)
    img[img == min_pixel_val] = 0.5
    return img

def randomize_background(img, min_range=0.0, max_range=0.5):
    min_pixel_val = torch.min(img)
    img[img == min_pixel_val] = (min_range - max_range) * torch.rand(img[img == min_pixel_val].shape, ) + max_range
    return img


    

