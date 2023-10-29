# The Vision Transformer (ViT)

This repository contains the code implementations of my masters research project, "Training Vision Transformers on Small Datasets". The project aims to explore and understand the application of transformers in computer vision tasks and serve as a foundation for further research and experimentation.


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Vision Transformer (ViT) is a pioneering architecture that adapts the transformer model, originally designed for natural language processing tasks, to image recognition tasks. This repository provides a basic implementation of the ViT model, along with training and evaluation scripts, allowing researchers and developers to experiment with and build upon the architecture.

<div style="display: flex; justify-content: center;">
  <img src="vit.gif" alt="solution" width="50%">
</div>

## Features

- Basic implementation of the Vision Transformer (ViT) model.
- Basic implementation of a Convolutional Neural Network (CNN) model - for comparison.
- Training pipeline with support for custom datasets.
- Evaluation scripts for image classification tasks.
- Easy-to-use configuration for model hyperparameters.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/pressi-g/vision-transformer-pytorch.git
   ```

   ```bash
   cd vision-transformer-pytorch
   ```

2. Create a virtual environment using conda:

   ```bash
   conda create -n pytorch-vit-env python=3.11
   ```

   ```bash
   conda activate pytorch-vit-env
   ```

3. Install PyTorch with M1/M2 support:

   ```bash
   conda install pytorch torchvision torchaudio -c pytorch-nightly
   ```

4. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

*Please note that these requirements are specific to the `2020 MacBook Pro with M1 chip`. You may need to review the dependencies and install the appropriate versions for your system.*

## Usage

Please refer to the notebooks titled `final_vit_<dataset>.ipynb` for the experiments for the dataset in question. 
These notebooks contain the finalised code to be refactored into scripts to streamline experiments.
This will happen in the near future. For now you may clone one of these notebooks (if you wish to to run your own experiments) and edit the cell with the config values: 

## Parameters

dataset: `str`
   - Options: MNIST, CIFAR10, CIFAR100

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 50
self_supervised_epochs = 20
fine_tune_epochs = 20
image_size = 28 if dataset == "MNIST" else 32 # upscale CIFAR10 and CIFAR100 images for better performance -> INVESTIGATE
patch_size = 7 if dataset == "MNIST" else 8 # keep the number of patches the same for all datasets to keep experiments controlled
projection_dim = 64
num_heads = 4
transformer_layers = 8 # depth
mlp_head_units = [2048, 1024]
patience_value = 10 if dataset == "MNIST" else 15 # default: 10 if not set
dropout=0.01 # Adjust as necessary
ema_decay = 0.996  # Adjust only if necessary

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or a pull request.

## License

This project is licensed under the [MIT License](LICENSE.md).