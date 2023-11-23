""" Utility functions for pytorch-vit. """

# Standard imports
import numpy as np
import os
import copy
import time
import random

# Data science imports
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Other imports
from tqdm import tqdm

# Local imports
from config import *


def set_device():
    """
    Set device: either Cuda (GPU), MPS (Apple Silicon GPU), or CPU
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def set_seed(seed: int = 42):
    """
    Set seed for reproducibility.

    Parameters:
        seed (int): Seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_loader(
    dataset: str,
    batch_size: int,
    image_size: int,
    data_use: int = 100,
    augment: bool = True,
):
    """
    Function that takes in a dataset and returns train and test dataloaders along with the hyperparameters
    associated with the dataset.

    Parameters:
        dataset (str): Name of dataset to load. Options: MNIST, CIFAR10, CIFAR100
        batch_size (int): Batch size for dataloaders
        augment (bool): Whether to augment training data or not

    Returns:
        train_loader (DataLoader): Dataloader for training set
        val_loader (DataLoader): Dataloader for validation set
        test_loader (DataLoader): Dataloader for test set
        num_classes (int): Number of classes in dataset
        image_size (int): Size of image in dataset
        batch_size (int): Batch size for dataloaders
    """

    base_train_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if dataset == "MNIST":
        base_train_transforms.append(transforms.Normalize((0.5,), (0.5,)))
    elif dataset == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        base_train_transforms.append(transforms.Normalize(mean, std))
    elif dataset == "CIFAR100":
        mean, std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        base_train_transforms.append(transforms.Normalize(mean, std))

    # Additional augmentations for CIFAR10 and CIFAR100
    additional_transforms = []
    if augment:
        additional_transforms = [
            transforms.Resize(
                (image_size, image_size)
            ),  # resizing capabilities if needed. Currently not used for our experiments
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(3.6),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
        ]

    dataset_config = {
        "MNIST": {
            "dataset_cls": datasets.MNIST,
            "num_classes": 10,
            "default_image_size": 28,
            "batch_size": batch_size,  # Use the batch_size parameter
            "train_transform": transforms.Compose(base_train_transforms),
            "test_transform": transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        },
        "CIFAR10": {
            "dataset_cls": datasets.CIFAR10,
            "num_classes": 10,
            "default_image_size": 32,
            "batch_size": batch_size,
            "train_transform": transforms.Compose(
                additional_transforms + base_train_transforms
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    ),
                ]
            ),
        },
        "CIFAR100": {
            "dataset_cls": datasets.CIFAR100,
            "num_classes": 100,
            "default_image_size": 32,
            "batch_size": batch_size,
            "train_transform": transforms.Compose(
                additional_transforms + base_train_transforms
            ),
            "test_transform": transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                    ),
                ]
            ),
        },
    }

    # Ensure dataset is valid
    if dataset not in dataset_config:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported datasets are: {', '.join(dataset_config.keys())}"
        )

    # Access config
    cfg = dataset_config[dataset]

    # Print loading info
    print(f"Loading {dataset} dataset...")

    # Load data
    dataset_cls = cfg["dataset_cls"]

    data_dir = "../data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_dataset_full = dataset_cls(
        f"{data_dir}{dataset}_data/",
        train=True,
        download=True,
        transform=cfg["train_transform"],
    )
    # decrease the amount of training data to data_use percentage
    train_dataset_full, _ = train_test_split(
        train_dataset_full,
        train_size=data_use / 100,
        random_state=42,
        shuffle=True,
        stratify=train_dataset_full.targets,
    )

    test_dataset = dataset_cls(
        f"{data_dir}{dataset}_data/",
        train=False,
        download=True,
        transform=cfg["test_transform"],
    )
    # decrease the amount of test data to data_use value
    test_dataset, _ = train_test_split(
        test_dataset,
        train_size=data_use / 100,
        random_state=42,
        shuffle=True,
        stratify=test_dataset.targets,
    )

    # Split training dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size]
    )

    # Compute optimal number of workers
    num_workers = min(4, os.cpu_count())

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        cfg["num_classes"],
        cfg["default_image_size"],
    )


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    model_name,
    epochs=10,
    scheduler=None,
    patience=None,
):
    """
    Function that trains the model.

    Parameters:
        model (nn.Module): Model to train
        train_loader (DataLoader): Dataloader for training set
        val_loader (DataLoader): Dataloader for validation set
        criterion (nn.Module): Loss function
        optimizer (nn.Module): Optimizer
        device (str): Device to use
        epochs (int): Number of epochs
        scheduler (nn.Module): Learning rate scheduler
        patience (int): Number of epochs to wait before early stopping
        model_name (str): Name of model

    Returns:
        model (nn.Module): Trained model
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_acc (list): Training accuracy
        val_acc (list): Validation accuracy
    """

    # Set model to training mode
    model.train()

    # Initialize lists to store losses and accuracy
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    best_val_loss = np.inf

    epochs_without_improvement = 0  # for early stopping

    # Record the start time for training
    start_time = time.time()

    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_val_loss = 0
        epoch_val_acc = 0

        # Training
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            epoch_train_acc += torch.sum(preds == labels.data)

        # Validation
        model.eval()  # <-- set model to eval mode for validation
        for inputs, labels in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            epoch_val_acc += torch.sum(preds == labels.data)
        model.train()  # <-- set model back to train mode

        # Average metrics
        epoch_train_loss /= len(train_loader.dataset)
        epoch_train_acc = epoch_train_acc.float() / len(train_loader.dataset)
        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_acc = epoch_val_acc.float() / len(val_loader.dataset)

        if scheduler:
            scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0  # reset the count
        else:
            epochs_without_improvement += 1

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Train Acc: {epoch_train_acc:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        # Early stopping
        if patience and epochs_without_improvement == patience:
            print("Early stopping due to no improvement in validation loss.")
            break

    # Record the end time for training
    end_time = time.time()
    # Calculate the total training time
    total_time_seconds = end_time - start_time
    if total_time_seconds < 60:
        print(f"Total training time: {total_time_seconds:.2f} seconds")
    else:
        total_time_minutes = total_time_seconds / 60
        print(f"Total training time: {total_time_minutes:.2f} minutes")

    model.load_state_dict(best_model_wts)
    # save trained model

    save_dir = "../trained_models/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), f"{save_dir}{model_name}.pth")

    return model, train_losses, val_losses, train_acc, val_acc


def create_directory(directory):
    """
    Function that creates a directory if it does not exist.

    Parameters:
        directory (str): Directory to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_views(x, global_ratio=0.5, local_ratio=0.5, strategy="random"):
    """
    Function that creates global and local views of an image.

    Parameters:
        x (Tensor): Input image tensor of shape (B, C, H, W)
        global_ratio (float): Fraction of the image to be used for global views
        local_ratio (float): Fraction of the image to be used for local views
        strategy (str): Strategy to use for local views. Options: "random", "central", "jitter"

    Returns:
        global_views (list): List of global views
        local_views (list): List of local views
    """

    B, C, H, W = x.size()

    global_views = []
    local_views = []

    # Create 2 Global Views
    for _ in range(2):
        global_view = F.interpolate(x, scale_factor=global_ratio)
        global_view = F.interpolate(global_view, size=(H, W))
        global_views.append(global_view)

    # Create 8 Local Views
    for _ in range(8):
        new_H, new_W = int(H * local_ratio), int(W * local_ratio)

        if strategy == "random":
            top = random.randint(0, H - new_H)
            left = random.randint(0, W - new_W)
        elif strategy == "central":
            top = (H - new_H) // 2
            left = (W - new_W) // 2
        elif strategy == "jitter":
            crop_ratio = local_ratio + random.uniform(-0.1, 0.1)  # Adding jitter
            new_H_jitter, new_W_jitter = int(H * crop_ratio), int(W * crop_ratio)
            top = random.randint(0, H - new_H_jitter)
            left = random.randint(0, W - new_W_jitter)
        else:
            raise ValueError(f"Unknown local view strategy: {strategy}")

        local_view = x[:, :, top : top + new_H, left : left + new_W]
        local_view_resized = F.interpolate(local_view, scale_factor=local_ratio)
        local_view_resized = F.interpolate(local_view_resized, size=(new_H, new_W))
        local_views.append(local_view_resized)

    return global_views, local_views


def center_and_sharpen(features, sharpen_value=0.5):
    """
    Center and sharpen the features.

    Parameters:
        features (torch.Tensor): The features tensor.
        sharpen_value (float): Value used for sharpening.

    Returns:
        torch.Tensor: Centered and sharpened features.
    """
    # Centering
    centered_features = features - features.mean(dim=0)

    # Sharpening
    sharpened_features = torch.nn.functional.softmax(
        centered_features / sharpen_value, dim=1
    )

    return sharpened_features


class SelfSupervisedLoss(nn.Module):
    """
    Implements the self-supervised loss for Vision Transformer as described in
    "How to Train Vision Transformers on Small-scale Datasets".

    This loss combines both global and local views. Specifically,
    it computes a loss based on the negative sum of the elementwise product
    of the student's outputs and the logarithm of the teacher's outputs.

    Attributes:
        None

    Methods:
        forward(student_global, teacher_global, student_locals, teacher_locals): Computes the self-supervised loss.
    """

    def __init__(self, temperature=0.1):
        super(SelfSupervisedLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_global, teacher_global, student_locals, teacher_locals):
        """
        Computes the self-supervised loss.

        Parameters:
        - student_global (torch.Tensor): Global view output from the student network.
        - teacher_global (torch.Tensor): Global view output from the teacher network.
        - student_locals (List[torch.Tensor]): List of local view outputs from the student network.
        - teacher_locals (List[torch.Tensor]): List of local view outputs from the teacher network.

        Returns:
        - loss (torch.Tensor): Computed self-supervised loss combining both global and local views.
        """

        # Temperature-scaled cross-entropy loss
        teacher_global = torch.nn.functional.log_softmax(
            teacher_global / self.temperature, dim=-1
        )
        student_global = torch.nn.functional.softmax(
            student_global / self.temperature, dim=-1
        )

        # Compute the global view loss component
        global_loss = -torch.mean(torch.sum(student_global * teacher_global, dim=-1))

        # Compute the local view loss component
        teacher_locals = [
            torch.nn.functional.log_softmax(t_local / self.temperature, dim=-1)
            for t_local in teacher_locals
        ]
        student_locals = [
            torch.nn.functional.softmax(s_local / self.temperature, dim=-1)
            for s_local in student_locals
        ]

        local_losses = [
            -torch.mean(torch.sum(s_local * t_local, dim=-1))
            for s_local, t_local in zip(student_locals, teacher_locals)
        ]

        # Average the local losses
        total_local_loss = sum(local_losses) / len(local_losses)

        # Combine the losses
        loss = global_loss + total_local_loss
        return loss


def self_supervised_training(
    student,
    teacher,
    optimizer,
    ema_updater,
    epochs,
    train_loader,
    val_loader,
    scheduler,
    device,
    criterion,
):
    """
    Train Vision Transformer in a self-supervised manner.

    Parameters:
        student (nn.Module): Student Vision Transformer model
        teacher (nn.Module): Teacher Vision Transformer model
        optimizer (nn.Module): Optimizer
        ema_updater (EMA): EMA updater
        epochs (int): Number of epochs
        train_loader (DataLoader): Training set dataloader
        val_loader (DataLoader): Validation set dataloader
        scheduler (nn.Module): Learning rate scheduler
        device (str): Device to use
        criterion (nn.Module): Loss function

    Returns:
        None
    """

    student.train()
    teacher.eval()  # Teacher always in eval mode
    loss_fn = SelfSupervisedLoss()
    best_val_loss = float("inf")  # Initialize best validation loss to infinity§

    # create save directory if it doesn't exist
    save_dir = "../trained_models/dino/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Start timer
    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        for images, _ in tqdm(
            train_loader, desc="Training"
        ):  # We don't need labels in this phase
            images = images.to(device)

            # Create views
            global_views, local_views = create_views(images)

            total_view_loss = 0
            for i in range(len(global_views)):
                global_view = global_views[i].to(device)
                local_view = local_views[i].to(device)

                # Forward pass
                with torch.no_grad():
                    teacher_features = teacher(global_view)
                    teacher_features = center_and_sharpen(
                        teacher_features
                    )  # Apply centering and sharpening
                    teacher_features = F.normalize(teacher_features, dim=1)

                student_global_features, student_local_features = student(
                    global_view
                ), student(local_view)
                student_global_features = F.normalize(student_global_features, dim=1)

                # Loss computation for this view
                loss = loss_fn(
                    student_global_features,
                    teacher_features,
                    student_local_features,
                    teacher_features,
                )
                total_view_loss += loss

            # Average the loss for all views
            avg_view_loss = total_view_loss / len(global_views)
            total_loss += avg_view_loss.item()

            # Backpropagate based on avg_view_loss
            optimizer.zero_grad()
            avg_view_loss.backward()
            optimizer.step()

            # Update teacher with EMA of student weights
            ema_updater.apply()

        # Average loss for the whole epoch
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {avg_loss:.4f}")

        # Use the validation set to adapt the learning rate
        with torch.no_grad():
            val_loss = 0.0
            student.eval()  # Switch student to evaluation mode

            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            print(
                f"Validation Loss after Epoch [{epoch+1}/{epochs}]: {avg_val_loss:.4f}"
            )

            # Save the model if the validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    student.state_dict(),
                    f"{save_dir}{dataset}_{data_use}_self_supervised_best_student_weights.pth",
                )

                print(
                    "Saved improved model with validation loss {:.4f}".format(
                        avg_val_loss
                    )
                )

        student.train()  # Switch student back to training mode after validation

    # Record the end time for training
    end_time = time.time()
    # Calculate the total training time
    total_time_seconds = end_time - start_time
    if total_time_seconds < 60:
        print(f"Total training time: {total_time_seconds:.2f} seconds")
    else:
        total_time_minutes = total_time_seconds / 60
        print(f"Total training time: {total_time_minutes:.2f} minutes")

    # At the end of training, you might want to restore the student's original weights
    ema_updater.restore()
    print("Self-supervised training complete!")


def supervised_training(
    student,
    optimizer,
    criterion,
    num_epochs,
    train_loader,
    val_loader,
    scheduler,
    device,
):
    """
    Fine-tune the Vision Transformer in a supervised manner.

    Parameters:
    - student (nn.Module): The student model.
    - optimizer (torch.optim.Optimizer): Optimizer for the student model.
    - criterion (nn.Module): Loss function.
    - num_epochs (int): Number of training epochs.
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - scheduler: Learning rate scheduler.
    - device (torch.device): Device to which tensors will be moved.

    Returns:
    None
    """

    best_val_loss = float("inf")  # Set the best validation loss to infinity

    # create save directory if it doesn't exist
    save_dir = "../trained_models/dino/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Start timer
    start_time = time.time()

    for epoch in range(num_epochs):
        student.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = student(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        # Use the validation set to adapt the learning rate
        with torch.no_grad():
            val_loss = 0.0
            student.eval()

            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            print(
                f"Validation Loss after Epoch [{epoch+1}/{num_epochs}]: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    student.state_dict(),
                    f"{save_dir}{dataset}_{data_use}_supervised_best_student_weights.pth",
                )

    # Record the end time for training
    end_time = time.time()
    # Calculate the total training time
    total_time_seconds = end_time - start_time
    if total_time_seconds < 60:
        print(f"Total training time: {total_time_seconds:.2f} seconds")
    else:
        total_time_minutes = total_time_seconds / 60
        print(f"Total training time: {total_time_minutes:.2f} minutes")

    print("Supervised fine-tuning complete!")


def initialize_optimizer_scheduler_criterion(
    model, learning_rate, weight_decay, patience_value
):
    """
    Initialize optimizer and scheduler.

    Parameters:
        model (nn.Module): Model
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        patience_value (int): Patience value for scheduler

    Returns:
        optimizer (nn.Module): Optimizer
        scheduler (nn.Module): Scheduler
    """

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=patience_value, factor=0.5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion


def plot_losses(train_losses, val_losses, plot_dir, model_name, dataset, data_use):
    """
    Plot training and validation losses.

    Parameters:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        plot_dir (str): Directory to save plots
        model_name (str): Name of model
        dataset (str): Name of dataset
        data_use (int): Percentage of data used

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    plt.title(f"{dataset} Basic ViT - {data_use}% Data: Loss")
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{plot_dir}{model_name}_basic_vit_loss.pdf", format="pdf")
    plt.show()


def plot_accuracies(train_acc, val_acc, plot_dir, model_name, dataset, data_use):
    """
    Plot training and validation accuracies.

    Parameters:
        train_acc (list): Training accuracies
        val_acc (list): Validation accuracies
        plot_dir (str): Directory to save plots
        model_name (str): Name of model
        dataset (str): Name of dataset
        data_use (int): Percentage of data used

    Returns:
        None
    """

    plt.figure(figsize=(6, 4))
    plt.title(f"{dataset} Basic ViT- {data_use}% Data: Accuracy")
    train_acc = [i.cpu() for i in train_acc]  # move list to cpu
    val_acc = [i.cpu() for i in val_acc]
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{plot_dir}{model_name}_basic_vit_acc.pdf", format="pdf")
    plt.show()


def evaluate_model(model, test_loader, device, dataset, plot_dir, model_name):
    """
    Evaluate model on test set.

    Parameters:
        model (nn.Module): Model
        test_loader (DataLoader): Test set dataloader
        device (str): Device to use
        dataset (str): Name of dataset
        plot_dir (str): Directory to save plots
        model_name (str): Name of model

    Returns:
        None
    """
    # Set model to eval mode
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            preds.extend(pred.cpu().numpy())
            labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(labels, preds)
    print(f"Test accuracy: {accuracy:.4f}")

    if dataset != "CIFAR100":
        print(classification_report(labels, preds))
        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{dataset} Basic ViT: Confusion Matrix (in %)")
        plt.savefig(f"{plot_dir}{model_name}_basic_vit_cm.pdf", format="pdf")
        plt.show()
