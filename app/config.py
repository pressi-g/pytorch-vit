""" Configuration file for ViT and CNN model training and evaluation.  """

# Editable parameters
dataset = "MNIST"  # <-- change this parameter only! Options: MNIST, CIFAR10, CIFAR100
data_use = 75  # <-- change this parameter for scaling dataset in percentage!
learning_rate = 0.002
weight_decay = 0.0001
batch_size = 256
num_epochs = 50
self_supervised_epochs = 20
fine_tune_epochs = 20
image_size = (
    28 if dataset == "MNIST" else 32
)
patch_size = (
    7 if dataset == "MNIST" else 8
)
projection_dim = 64
num_heads = 4
transformer_layers = 8  # depth
mlp_head_units = [2048, 1024]
patience_value = 10 if dataset == "MNIST" else 15  # default: 10 if not set
dropout = 0.01  # Adjust as necessary
ema_decay = 0.999  # Adjust only if necessary
drop = 0.2  # Adjust only if necessary

# Fixed parameters
model_name = f"{dataset}_{data_use}_classifier"
in_channels = 1 if dataset == "MNIST" else 3
num_classes = 100 if dataset == "CIFAR100" else 10
transformer_units = [projection_dim * 2, projection_dim]
num_patches = (image_size // patch_size) ** 2

