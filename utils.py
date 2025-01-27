import os
import torch
import shutil
import random
import imageio
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
import config


def clear_directories():
    """
    Deletes all directories specified in the configuration file.
    This is useful for clearing previous training outputs.
    """
    for directory in config.DIRECTORIES:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"{directory}/ deleted successfully!")


def save_checkpoint(type, epoch, model, optimizer, dir=config.MODEL_DIR):
    """
    Saves the model and optimizer states as a checkpoint.

    Args:
        type (str): The type of model to save ('critic' or 'generator').
        epoch (int): The current epoch, used to name the checkpoint.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save states.
        dir (str, optional): Directory to store the checkpoint. Defaults to config.MODEL_DIR.
    """
    print("Saving checkpoint......")
    os.makedirs(dir, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filepath = f"{dir}/{type}_{epoch}.pth"
    torch.save(checkpoint, filepath)
    print("Checkpoint saved successfully.")


def load_checkpoint(type, epoch, model, optimizer, dir=config.MODEL_DIR, learning_rate=config.LEARNING_RATE):
    """
    Loads a saved model checkpoint.

    Args:
        type (str): The type of model to load ('critic' or 'generator').
        epoch (int): Load the model from which epoch.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): The optimizer to restore states from the checkpoint.
        dir (str, optional): Directory where the checkpoint is stored. Defaults to config.MODEL_DIR.
        learning_rate (float, optional): Learning rate to be set after loading the optimizer state. Defaults to config.LEARNING_RATE.
    """
    checkpoint_path = os.path.join(dir, f"{type}_{epoch}.pth")

    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Checkpoint file '{checkpoint_path}' not found. Falling back without loading checkpoint.")
        return

    print("Loading checkpoint......")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    print("Checkpoint loaded successfully.")


def plot_training_losses(epoch, critic_losses, generator_losses, save_dir=config.ASSETS_DIR, filename="gan_loss"):
    """
    Plots and saves the loss curves for the critic and generator during training.

    Args:
        epoch (int): Current epoch number for naming the saved plot.
        critic_losses (list): List of critic loss values over epochs.
        generator_losses (list): List of generator loss values over epochs.
        save_dir (str, optional): Directory where plots will be saved. Defaults to config.ASSETS_DIR.
        filename (str, optional): Base name of the saved plot file. Defaults to "gan_loss".
    """
    plt.figure(figsize=(10, 5))
    plt.plot(critic_losses, label="Critic Loss")
    plt.plot(generator_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Loss")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{filename}_{epoch}.png")

    plt.show()


def log_metrics_to_tensorboard(writer, critic_loss, generator_loss, real_images, fake_images, step):
    """
    Logs loss values and generated images to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        critic_loss (float): Current critic loss value.
        generator_loss (float): Current generator loss value.
        real_images (Tensor): Batch of real images.
        fake_images (Tensor): Batch of generated images.
        step (int): Current training step for logging.
    """
    writer.add_scalar("Critic Loss", critic_loss, global_step=step)
    writer.add_scalar("Generator Loss", generator_loss, global_step=step)
    with torch.no_grad():
        img_grid_real = torchvision.utils.make_grid(real_images[:32], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake_images[:32], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=step)
        writer.add_image("Fake", img_grid_fake, global_step=step)


def save_generated_images(epoch_folder, generator, alpha, step, epoch, save_dir=config.IMAGE_DIR, fixed_noise=config.FIXED_NOISE):
    """
    Generates and saves images from the generator model.

    Args:
        epoch_folder (str): Folder name (e.g., epoch number) for saving images.
        generator (torch.nn.Module): Generator model.
        alpha (float): Alpha value for progressive GAN training.
        step (int): Current training step.
        epoch (int): Current epoch number for file naming.
        save_dir (str, optional): Directory to save images. Defaults to config.IMAGE_DIR.
        fixed_noise (Tensor, optional): Fixed input noise for consistency across epochs.
    """
    save_path = os.path.join(save_dir, epoch_folder)
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        fake = generator(fixed_noise, alpha, step)
        img_grid = torchvision.utils.make_grid(fake, normalize=True)
        image_path = os.path.join(save_path, f"epoch_{epoch}.png")
        torchvision.utils.save_image(img_grid, image_path)


def create_gif(epoch_folder, save_dir=config.ASSETS_DIR, image_dir=config.IMAGE_DIR, filename="gan_training"):
    """
    Creates a GIF from the saved images of a particular epoch.

    Args:
        epoch_folder (str): Folder containing images for the GIF.
        save_dir (str, optional): Directory to save the generated GIF. Defaults to config.ASSETS_DIR.
        image_dir (str, optional): Directory containing the images. Defaults to config.IMAGE_DIR.
        filename (str, optional): Base name for the GIF file. Defaults to "gan_training".
    """
    images_path = os.path.join(image_dir, epoch_folder)
    if not os.path.exists(images_path):
        print(f"Warning: No images found in {images_path}.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".png")])
    if not image_files:
        print(f"Warning: No PNG images in {images_path}. Skipping GIF creation.")
        return

    gif_images = [imageio.imread(img) for img in image_files]
    gif_path = os.path.join(save_dir, f"{filename}_{epoch_folder}.gif")
    imageio.mimsave(gif_path, gif_images, fps=5)
    print(f"GIF saved at: {gif_path}")


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """
    Computes the gradient penalty used in Progressive Growing of GANs (ProGAN).
    
    In ProGAN, gradient penalty helps stabilize training during progressive layer 
    transitions by enforcing smooth interpolation between real and fake images.

    Args:
        critic (nn.Module): The critic (discriminator) model, which evaluates real and fake images.
        real (torch.Tensor): A batch of real images from the dataset, shape (B, C, H, W).
        fake (torch.Tensor): A batch of generated (fake) images, shape (B, C, H, W).
        alpha (float): Blending factor (0 to 1) controlling the fade-in of new layers 
                       during progressive training.
        train_step (int): The current training step, used for progressive growing logic.
        device (str, optional): The device to run computations on ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        torch.Tensor: The computed gradient penalty scalar value.
    """
    
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    # Create interpolated images by linearly blending real and fake images using beta
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)  # Enable gradient tracking for gradient penalty calculation

    # Forward pass: Compute critic scores for the interpolated images
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Compute gradients of critic scores w.r.t. interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),  # Ensures correct gradient computation
        create_graph=True,  # Required for second-order gradient calculations
        retain_graph=True,  # Retain computational graph for further use
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)  # Reshape gradients to (B, -1) for easier norm computation
    gradient_norm = gradient.norm(2, dim=1)  # Compute the L2 norm (Euclidean norm) for each sample in the batch
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2) # Compute the gradient penalty: Encouraging ||gradient|| ≈ 1 for smooth training
    
    return gradient_penalty