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
- Basic implementation of a Convolutional Neural Network (CNN) model.
- Basic implementation of a Knowledge Distillation with No Labels model (DINO).
- Training pipeline with support for custom datasets.
- Easy-to-use configuration for model hyperparameters.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/pressi-g/pytorch-vit
   ```

   ```bash
   cd pytorch-vit
   ```

2. Create a virtual environment using conda:

   ```bash
   conda create -n pytorch-vit-env python=3.11
   ```

   ```bash
   conda activate pytorch-vit-env
   ```

3. **Optional**: Install PyTorch with M1/M2 support:

   ```bash
   conda install pytorch torchvision torchaudio -c pytorch-nightly
   ```

4. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

*Please note that these requirements are specific to the `2020 MacBook Pro with M1 chip`. You may need to review the dependencies and install the appropriate versions for your system.*

## Usage

To run your own experiments please clone the notebook with model you'd like to use. Change the config values in the second cell and run the notebook. Alternatively, you may view the experiments and its logs.

These notebooks contain the finalised code to be refactored into scripts to streamline experiments. All modules are reusable.

Refactoring into scripts will happen in the near future.

## Parameters

```
dataset: `str`
   - Options: MNIST, CIFAR10, CIFAR100
leanring_rate: `float`
   - Default: 0.002
weight_decay: `float`
   - Default: 0.0001
batch_size: `int`
   - Default: 256
num_epochs: `int`
   - Default: 50
self_supervised_epochs: `int`
   - Default: 20
fine_tune_epochs: `int`
   - Default: 20
image_size: `int`
   - Default: 28 if dataset == "MNIST" else 32
patch_size: `int`
   - Default: 7 if dataset == "MNIST" else 8
projection_dim: `int`
   - Default: 64
num_heads: `int`
   - Default: 4
transformer_layers: `int`
   - Default: 8
mlp_head_units: `list`
   - Default: [2048, 1024]
patience_value: `int`
   - Default: 10 if not set
dropout: `float`
   - Default: 0.01
ema_decay: `float`
   - Default: 0.996
```

- dataset: The dataset to use for training and evaluation. Currently supports MNIST, CIFAR10 and CIFAR100.
- learning_rate: The learning rate to use for training.
- weight_decay: The weight decay to use for training.
- batch_size: The batch size to use for training.
- num_epochs: The number of epochs to use for training.
- self_supervised_epochs: The number of epochs to use for self-supervised pre-training.
- fine_tune_epochs: The number of epochs to use for fine-tuning.
- image_size: The image size to use for training and evaluation.
- patch_size: The patch size to use for training and evaluation.
- projection_dim: The projection dimension to use for training and evaluation.
- num_heads: The number of heads to use for training and evaluation.
- transformer_layers: The number of transformer layers to use for training and evaluation.
- mlp_head_units: The number of units to use for the MLP head.
- patience_value: The patience value to use for early stopping.
- dropout: The dropout value to use for training and evaluation.
- ema_decay: The exponential moving average decay to use for training and evaluation.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or a pull request.

## License

This project is licensed under the [MIT License](LICENSE.md).