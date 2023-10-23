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

1. Prepare your dataset:
   - Download and preprocess your dataset.
   - Update the dataset paths and configurations in `config.yaml`.

2. Train the Vision Transformer model:

   ```bash
   python train.py --config config.yaml
   ```

3. Evaluate the trained model:

   ```bash
   python evaluate.py --config config.yaml --checkpoint /path/to/checkpoint.pth
   ```

## Results

TBA


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or a pull request.

## License

This project is licensed under the [MIT License](LICENSE.md).