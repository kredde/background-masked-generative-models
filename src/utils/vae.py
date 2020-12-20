from sklearn.manifold import TSNE
import seaborn as sns
import torch
import matplotlib.pyplot as plt

def generate_images(model, test_data):
    columns = 4
    rows = 4
    fig = plt.figure(figsize=(8, 8))
    i = 1
    for img in iter(test_data):
        if i <= (columns * rows) * 2:
            fig.add_subplot(rows * 2, columns * 2, i)
            img = img[0][0]
            img = img.repeat(1, 3, 1, 1)
            out = model(img.cuda())
            reconstructed_img = out[0, 0, :, :]
            plt.imshow(reconstructed_img.detach().cpu().numpy(), cmap="gray")
            plt.xticks([])
            plt.yticks([])
        
            fig.add_subplot(rows * 2, columns * 2, i+1)
            plt.imshow(img[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
            plt.xticks([])
            plt.yticks([])
        i += 2
    plt.show()
    



    

