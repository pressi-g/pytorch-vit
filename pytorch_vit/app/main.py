""" Main script to train a CNN or ViT model. """

# Standard imports
import argparse

# Machine learning imports
from torchsummary import summary

# Local imports
from models.cnn import CNN
from models.vit.vit import VisionTransformer, VisionTransformerDPE
from utils import *


def main(args):
    # Set the device
    device = set_device()

    # Set the random seed
    set_seed()
    
    # Load the data
    train_loader, val_loader, test_loader, num_classes, default_image_size = data_loader(dataset, batch_size, image_size, data_use, augment=True)
    
    # Initialize the chosen model
    if args.model_type == 'vit':

        model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=projection_dim,
            depth=transformer_layers,
            heads=num_heads,
            mlp_dim=mlp_head_units[0],
            dropout=dropout,
            num_classes=num_classes,
        )

        # Print model summary
        summary(model, input_size=(in_channels, image_size, image_size))
        # Send model to device
        model = model.to(device)

        # Initialise optimizer and scheduler
        optimizer, scheduler, criterion = initialize_optimizer_scheduler_criterion(model, args.learning_rate, args.weight_decay, args.patience_value)

        # Train model
        model, train_losses, val_losses, train_acc, val_acc = train(model, train_loader, val_loader, criterion, optimizer, device, model_name, epochs=num_epochs, scheduler=scheduler, patience=patience_value)
    

    elif args.model_type == 'vit_dpe':
            
            model = VisionTransformerDPE(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=projection_dim,
                depth=transformer_layers,
                heads=num_heads,
                mlp_dim=mlp_head_units[0],
                dropout=dropout,
                num_classes=num_classes,
            )
    
            # Print model summary
            summary(model, input_size=(in_channels, image_size, image_size))
            # Send model to device
            model = model.to(device)
    
         
            # Initialise optimizer and scheduler
            optimizer, scheduler = initialize_optimizer_scheduler_criterion(model, args.learning_rate, args.weight_decay, args.patience_value)
    
            # Train model
            model, train_losses, val_losses, train_acc, val_acc = train(model, train_loader, val_loader, criterion, optimizer, device, model_name, epochs=num_epochs, scheduler=scheduler, patience=patience_value)


    elif args.model_type == 'cnn':
        model = CNN(dataset, drop)
        # Print model summary
        summary(model, input_size=(in_channels, default_image_size, default_image_size))
        # Send model to device
        model = model.to(device)


        # Initialise optimizer and scheduler
        optimizer, scheduler = initialize_optimizer_scheduler_criterion(model, args.learning_rate, args.weight_decay, args.patience_value)

        # Train model
        model, train_losses, val_losses, train_acc, val_acc = train(model, train_loader, val_loader, criterion, optimizer, device, model_name, epochs=num_epochs, scheduler=scheduler, patience=patience_value)
        

    else:
        raise ValueError("Invalid model type provided. Choose 'cnn' or 'vit or vit_dpe'.")
    
    if args.evaluate:
        # Code to evaluate the model goes here
        print("Evaluating the model on the test dataset...")
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # plot the training and validation loss
        plot_losses(train_losses, val_losses, model_name)

        # plot the training and validation accuracy
        plot_accuracies(train_acc, val_acc, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN or ViT model.")
    parser.add_argument('--model_type', type=str, choices=['cnn', 'vit', 'vit_dpe'], required=True,
                        help='Type of model to train: "cnn" or "vit" or "vit_dpe".')
    parser.add_argument('--evaluate', type=str, choices=["True", "False"], required=True, help='Evaluate the model on the test set after training.')
    args = parser.parse_args()
    main(args)
