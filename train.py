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
    plot_losses,
    save_fake_images,
    create_gif,
    gradient_penalty,
    plot_to_tensorboard,
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


def get_epoch(model):
    epoch = input(f"What epoch do you want to load the {model} model from: ").strip()

    if not epoch.isdigit():
        print("Invalid input. Defaulting to epoch 0.")
        epoch = 0
    else:
        epoch = int(epoch)

    return epoch


def get_loader(image_size):
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


def train_fn(
        critic,
        generator,
        opt_critic,
        opt_gen,
        scaler_critic,
        scaler_gen,
        writer,
        tensorboard_step,
        step,
        alpha,
        loader,
        dataset,
        epoch,
        epoch_num
):
    loop = tqdm(loader, leave=True)
    critic_loss_epoch = 0.0
    gen_loss_epoch = 0.0

    for batch_idx, (real, _) in enumerate(loop):
        """
        Critic Training

        The critic learns by:
        1. Getting real images -> predict 1
        2. Getting fake images -> predict 0
        3. Update weights to improve accuracy
        """

        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

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

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            gen_loss = -torch.mean(gen_fake)
            gen_loss_epoch += gen_loss.item()

        opt_gen.zero_grad()
        scaler_gen.scale(gen_loss).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            print(f"EPOCH[{epoch + 1}/{epoch_num}], "
                  f"BATCH[{batch_idx}/{len(loader)}]"
                  f"CRITIC LOSS: {critic_loss:.2f}, "
                  f"GEN LOSS: {gen_loss:.2f}"
                  )

            with torch.no_grad():
                fixed_fakes = generator(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5

                plot_to_tensorboard(
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


def train():
    clear_directories()

    critic = Critic(config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    generator = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)

    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(f"{config.LOG_DIR}/gan")

    if config.LOAD_MODEL:
        critic_epoch = get_epoch("critic")
        load_checkpoint("critic", critic_epoch, critic, opt_critic)

        gen_epoch = get_epoch("generator")
        load_checkpoint("generator", gen_epoch, generator, opt_gen)

    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))

    critic.train()
    generator.train()

    for epoch_num in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        image_size = 4 * 2 ** step
        loader, dataset = get_loader(image_size)
        print(f"Current image size: {image_size}")
        critic_losses, gen_losses = [], []

        for epoch in range(epoch_num):
            tensorboard_step, alpha, avg_critic_loss, avg_gen_loss = train_fn(
                critic,
                generator,
                opt_critic,
                opt_gen,
                scaler_critic,
                scaler_gen,
                writer,
                tensorboard_step,
                step,
                alpha,
                loader,
                dataset,
                epoch,
                epoch_num
            )

            critic_losses.append(avg_critic_loss)
            gen_losses.append(avg_gen_loss)

            if epoch % 10 == 0:
                if config.SAVE_MODEL:
                    save_checkpoint("critic", epoch, critic, opt_critic)
                    save_checkpoint("generator", epoch, generator, opt_gen)

            save_fake_images(str(epoch_num), generator, alpha, step, epoch)

        step += 1
        plot_losses(str(epoch_num), critic_losses, gen_losses)
        create_gif(str(epoch_num))


if __name__ == "__main__":
    train()