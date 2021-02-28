# Background-Masked Generative Models

This repository contains the code and scripts to run the models developed during the Large Scale Machine Learning Lab.

## Getting Started

1. Clone this repository
2. Install all dependencies `pip install -r requirements.txt`
3. Run the ood coco script in the `src/data/oodcoco` directory to generate the custom COCO persons dataset

## Executing models

Scripts to run the different models are located under `src/experiements`

## Results

### PixelCNN

#### Likelihood Heatmap Base Model
![Base PixelCNN COCO](https://raw.githubusercontent.com/kredde/background-masked-generative-models/master/doc/basecoco.png)

#### Likelihood Heatmap Pair Learning
![PairLearning PixelCNN COCO](https://raw.githubusercontent.com/kredde/background-masked-generative-models/master/doc/pixcoco.png)

### VAE with Variance
Reconstruction with different background values

#### Base VAE with Variance
![Base VAE Variance](https://raw.githubusercontent.com/kredde/background-masked-generative-models/master/doc/vaebase.png)

#### VAE with Variance, Custom Loss and Background Augmentation
![Custom Loss VAE Variance](https://raw.githubusercontent.com/kredde/background-masked-generative-models/master/doc/vae_custom.png)

