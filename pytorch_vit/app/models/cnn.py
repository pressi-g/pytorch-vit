""" Convolutional Neural Network (CNN) model for MNIST and CIFAR datasets. """

# Machine learning imports
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Simple CNN architecture.

    Parameters:
        dataset (str): Dataset
        drop (float): Dropout probability

    Returns:
        x (Tensor): Output of SimpleCNN
    """

    def __init__(self, dataset: str, drop: float):
        super(CNN, self).__init__()

        # Determine number of channels and classes based on the dataset
        in_channels = 1 if dataset == "MNIST" else 3
        num_classes = 100 if dataset == "CIFAR100" else 10

        # Pooling operation adjusted for dataset
        pool_kernel_size = 2
        pool_stride = 2  # Setting stride to 2 for both MNIST and CIFAR
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)

        if dataset == "MNIST":
            self.conv1 = nn.Conv2d(in_channels, 14, 3, padding=1)
            self.batch_norm1 = nn.BatchNorm2d(14)

            self.conv2 = nn.Conv2d(14, 28, 3, padding=1)
            self.batch_norm2 = nn.BatchNorm2d(28)

            self.conv3 = nn.Conv2d(28, 56, 3, padding=1)
            self.batch_norm3 = nn.BatchNorm2d(56)

            self.fc1 = nn.Linear(56 * 3 * 3, 512)

            self.dropout = nn.Dropout(drop)
            self.fc2 = nn.Linear(512, num_classes)
        else:
            self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
            self.batch_norm1 = nn.BatchNorm2d(16)

            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.batch_norm2 = nn.BatchNorm2d(32)

            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.batch_norm3 = nn.BatchNorm2d(64)

            self.fc1 = nn.Linear(64 * 4 * 4, 512)
            self.dropout = nn.Dropout(drop)
            self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))

        # Flattening
        x = x.view(x.size(0), -1)

        # Add first hidden layer, with relu activation function
        x = F.relu(self.fc1(x))

        # Add dropout layer
        x = self.dropout(x)

        # Add second hidden layer, with relu activation function
        x = self.fc2(x)

        return x
