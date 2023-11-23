""" Exponential Moving Average (EMA) for PyTorch models."""

# Machine learning imports
from torch.optim.lr_scheduler import CosineAnnealingLR


class EMA:
    """
    Class to update and maintain the Exponential Moving Average (EMA) of model parameters.

    Parameters:
        model (nn.Module): PyTorch model for which the EMA needs to be maintained.
        decay (float): Decay rate for the moving average. Value should be between 0 and 1.
                       Higher values give more importance to recent model parameters.

    Attributes:
        model (nn.Module): PyTorch model.
        decay (float): Decay rate for the moving average.
        shadow (dict): Dictionary to store the shadow weights.
        backup (dict): Dictionary to store the backup of the original model weights.
    """

    def __init__(self, model, ema_decay, num_epochs, optimizer):
        self.model = model
        self.decay_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            eta_min=0.001,
            last_epoch=-1,
            verbose=False,
        )
        self.initial_decay = ema_decay
        self.shadow = {}
        self.backup = {}

        # Initialize the shadow weights with the model parameters
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def apply(self):
        """
        Apply the EMA to the model weights. The model's weights are replaced by the shadow weights.
        """

        # Update decay based on the cosine schedule
        self.decay = (
            self.initial_decay
            + (1 - self.initial_decay) * self.decay_scheduler.get_last_lr()[0]
        )

        # Backup current model parameters
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data.clone()

        # Apply EMA to get the shadow weights
        for name, param in self.model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
            )
            param.data = self.shadow[name]

        # Step the scheduler for the next iteration
        self.decay_scheduler.step()

    def restore(self):
        """
        Restore the model's weights from the backup. This reverts the effects of the `apply` method.
        """
        # Restore model parameters from the backup
        for name, param in self.model.named_parameters():
            param.data = self.backup[name]
