from sklearn.manifold import TSNE
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_images(model, test_data, input=28, random_bg=False):
    columns = 3
    rows = 3
    fig = plt.figure(figsize=(10, 15))
    i = 1
    for img in iter(test_data):
        if i <= (columns * rows) * 3:
           
            img = img[0][0] if not random_bg else randomize_background(img[0][0])
            # img = img[0] # Just for constant images

            rec_mu, rec_var = model(img.cuda())
            std = torch.exp(rec_var / 2)
            normalized_std = normalize(std)
            p = torch.distributions.Normal(rec_mu, normalized_std)
            
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

def plot_loss_histogram(model, in_data, ood_data, input=28, random_bg=False):
    in_data_loss = []
    ood_data_loss = []
    for img in iter(in_data):
        img = img[0][0] if not random_bg else randomize_background(img[0][0])
        # img = img[0] # Just for constant images

        rec_mu, rec_var = model(img.cuda())
        std = torch.exp(rec_var / 2)
        normalized_std = normalize(std)
        p = torch.distributions.Normal(rec_mu, normalized_std)  

        img = torch.flatten(img, 1)
        recon_loss = torch.mean((rec_mu - img.cuda()) ** 2)
        in_data_loss.append(recon_loss.detach().cpu())
    
    for img in iter(ood_data):
        img = img[0][0] if not random_bg else randomize_background(img[0][0])
        # img = img[0] # Just for constant images

        rec_mu, rec_var = model(img.cuda())
        std = torch.exp(rec_var / 2)
        normalized_std = normalize(std)
        p = torch.distributions.Normal(rec_mu, normalized_std)  

        img = torch.flatten(img, 1)
        recon_loss = torch.mean((rec_mu - img.cuda()) ** 2)
        ood_data_loss.append(recon_loss.detach().cpu())
    
    plt.figure(figsize=(10,5))
    plt.hist(np.array(in_data_loss), 100, alpha=0.5, label='In Distribution Data')
    plt.hist(np.array(ood_data_loss), 150, alpha=0.5, label='OOD Data')
    plt.legend(loc='upper left')
    plt.xlabel('mse loss')
    plt.title('PixelCNN trained on MNIST')
    plt.show()

def normalize(data, min_range=0.0, max_range=1.0):
    min_pixel = torch.min(data)
    max_pixel = torch.max(data)

    min_range = min_range
    max_range = max_range

    scaled_data = ((data - min_pixel) / (max_pixel - min_pixel) * (max_range - min_range)) + min_range

    return scaled_data

def randomize_background(img, min_range=0.0, max_range=0.7):
    min_pixel_val = torch.min(img)
    img[img == min_pixel_val] = (min_range - max_range) * torch.rand(img[img == min_pixel_val].shape, ) + max_range
    return img


    

