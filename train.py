import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from math import log2
from tqdm import tqdm
import config
from model import Critic, Generator
from utils import (
    clear_directories,
    save_checkpoint,
    load_checkpoint,
    plot_training_losses,
    save_generated_images,
    create_gif,
    gradient_penalty,
    log_metrics_to_tensorboard,
)

"""
This is a PyTorch settings that optimizes performance when using CuDNN.

What is does:
When benchmark = True, PyTorch dynamically finds the fastest convolution algorithms for your model based on the input size.
If your input size stays the same across iterations, CuDNN caches the best algorithm and speeds up training.

When to use it:
✅ Use it when input sizes are fixed (e.g., images of the same size).
❌ Avoid it when input sizes change frequently (e.g., variable-sized sequences in NLP), as it will constantly search for the best algorithm, causing overhead.
"""
torch.backends.cudnn.benchmarks = True


def prompt_for_epoch(model):
    """Prompts the user for an epoch number to load a model from."""
    epoch = input(f"What epoch do you want to load the {model} model from: ").strip()

    if not epoch.isdigit():
        print("Invalid input. Defaulting to epoch 0.")
        return 0

    return int(epoch)


def prepare_dataloader(image_size):
    """Prepares the DataLoader for training, applying transformations and normalization."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.IMG_CHANNELS)],
                [0.5 for _ in range(config.IMG_CHANNELS)],
            ),
        ]
    )

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    return loader, dataset


def train_one_epoch(
        critic, generator, opt_critic, opt_gen,
        scaler_critic, scaler_gen, writer,
        tensorboard_step, step, alpha,
        loader, dataset, epoch, epoch_num
):
    """Trains the generator and critic for one epoch."""
    loop = tqdm(loader, leave=True)
    critic_loss_epoch, gen_loss_epoch = 0.0, 0.0

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        """
        Critic Training

        The critic learns by:
        1. Getting real images -> predict 1
        2. Getting fake images -> predict 0
        3. Update weights to improve accuracy
        """

        with torch.cuda.amp.autocast():
            fake = generator(noise, alpha, step)

            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)

            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)

            critic_loss = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
            )
            critic_loss_epoch += critic_loss.item()

        opt_critic.zero_grad()
        scaler_critic.scale(critic_loss).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        """
        Generator Training

        The generator learns by:
        1. Trying to fool the critic to predict 1 for fake images.
        """

        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            gen_loss = -torch.mean(gen_fake)
            gen_loss_epoch += gen_loss.item()

        opt_gen.zero_grad()
        scaler_gen.scale(gen_loss).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha (used for smooth fade-in)
        alpha += cur_batch_size / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            print(f"EPOCH[{epoch + 1}/{epoch_num}], BATCH[{batch_idx}/{len(loader)}], "
                  f"CRITIC LOSS: {critic_loss:.2f}, GEN LOSS: {gen_loss:.2f}")

            with torch.no_grad():
                fixed_fakes = generator(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                log_metrics_to_tensorboard(
                    writer,
                    critic_loss.item(),
                    gen_loss.item(),
                    real.detach(),
                    fixed_fakes.detach(),
                    tensorboard_step,
                )

            tensorboard_step += 1

        loop.set_postfix(gp=gp.item(), loss_critic=critic_loss.item())

    avg_critic_loss = critic_loss_epoch / len(loader)
    avg_gen_loss = gen_loss_epoch / len(loader)

    return tensorboard_step, alpha, avg_critic_loss, avg_gen_loss


def train_model():
    """Main function to train the GAN."""
    clear_directories()

    critic = Critic(config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    generator = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)

    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

    scaler_critic, scaler_gen = torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler()
    writer = SummaryWriter(f"{config.LOG_DIR}/gan")

    if config.LOAD_MODEL:
        load_checkpoint("critic", prompt_for_epoch("critic"), critic, opt_critic)
        load_checkpoint("generator", prompt_for_epoch("generator"), generator, opt_gen)

    tensorboard_step, step = 0, int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    critic.train(), generator.train()

    for epoch_num in config.PROGRESSIVE_EPOCHS[step:]:
        alpha, image_size = 1e-5, 4 * 2 ** step
        loader, dataset = prepare_dataloader(image_size)
        print(f"Current image size: {image_size}")
        critic_losses, gen_losses = [], []

        for epoch in range(epoch_num):
            tensorboard_step, alpha, avg_critic_loss, avg_gen_loss = train_one_epoch(
                critic, generator, opt_critic, opt_gen,
                scaler_critic, scaler_gen, writer,
                tensorboard_step, step, alpha,
                loader, dataset, epoch, epoch_num
            )

            critic_losses.append(avg_critic_loss)
            gen_losses.append(avg_gen_loss)

            if epoch % 10 == 0 and config.SAVE_MODEL:
                save_checkpoint("critic", epoch, critic, opt_critic)
                save_checkpoint("generator", epoch, generator, opt_gen)

            save_generated_images(str(epoch_num), generator, alpha, step, epoch)

        step += 1
        plot_training_losses(str(epoch_num), critic_losses, gen_losses)
        create_gif(str(epoch_num))


if __name__ == "__main__":
    train_model()