import torch
import torch.nn as nn
import torch.nn.functional as F


"""
This list determines how the number of channels changes at each layer of the generator and critic.

The first 5 layers keep the same number of channels.
After that, the number of channels reduces progressively.    
"""
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class WSConv2d(nn.Module):
    """
    Weight-Scaled Convolution
    
    What it does:
    - It normalizes weight updates to improve training stability.
    - It scales the input before applying convolution.
    - 'gain' helps to maintain proper weight magnitudes.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    """
    Pixel Normalization

    What it does:
    - It normalizes each feature vector (which is the pixel values in a layer) so that the network doesn't become unstable.
    - It prevents large values from dominating the learning process.
    - 'epsilon = 1e-8' helps to avoid division by zero.
    """
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    """
    Convolution Block

    What it does:
    - It applies 2 weight-scaled conv layers with LeakyReLU activation.
    - If 'use_pixelnorm=True', it normalizes outputs after each convolution.
    """
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    """
    Generator: Creates fake images.

    How it works:
    1. Starts with a 4x4 image using a transposed convolution.
    2. Progressively grows using prog_blocks.
    3. Converts feature maps to RGB using rgb_layers.
    4. Uses fade-in blend low-resolution images into higher resolution smoothly.
    """
    def __init__(self, z_dim, in_channels, img_channels):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks, self.rgb_layers = (nn.ModuleList([]), nn.ModuleList([self.initial_rgb]),)

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        """
        Smoothly blends two images: one from lower resolution (upscaled) and one from the new layer (generated).

        Args:
        - `alpha`: Blend factor (0 = only upscaled, 1 = only generated)
        - `upscaled`: Lower resolution image, resized to the new resolution
        - `generated`: Newly generated high-resolution image
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        """
        Generates an image from latent vector `x`.
        
        Args:
        - `alpha`: Fade-in factor (0 to 1) for smooth resolution transitions
        - `steps`: Number of resolution doubling steps
        """
        out = self.initial(x)  # Start with the 4x4 image

        if steps == 0:
            return self.initial_rgb(out)  # If first step, directly convert to RGB

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")  # Upscale previous output
            out = self.prog_blocks[step](upscaled)  # Apply convolutional block

        final_upscaled = self.rgb_layers[steps - 1](upscaled)  # Lower-resolution RGB
        final_out = self.rgb_layers[steps](out)  # Higher-resolution RGB
        return self.fade_in(alpha, final_upscaled, final_out)  # Blend images


class Critic(nn.Module):
    """
    Critic: Classifies real vs. fake.

    How it works:
    1. Starts with a high-resolution image and progressively downsamples it.
    2. Uses progressive blocks to extract features.
    3. The final block classifies whether the image is real or fake.
    4. Uses minibatch standard deviation to help detect mode collapse.
    """
    def __init__(self, in_channels, img_channels):
        super(Critic, self).__init__()
        
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),  # +1 for minibatch std
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha, downscaled, out):
        """
        Smooth transition between downscaled previous resolution and the new resolution.

        Args:
        - `alpha`: Blend factor (0 = only downscaled, 1 = only out)
        - `downscaled`: Lower-resolution features, reduced using avg pooling
        - `out`: Features from the new resolution step
        """
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """
        Adds a minibatch standard deviation channel to detect mode collapse.
        - Computes std across the batch, averages across channels.
        - Expands to match feature map size and concatenates to input.
        """
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        """
        Classifies an image as real or fake.

        Args:
        - `alpha`: Fade-in blending factor
        - `steps`: Number of resolution doubling steps
        """
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))  # Convert image to feature map

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)  # Flatten output

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))  # Downscale image
        out = self.avg_pool(self.prog_blocks[cur_step](out))  # Apply convolution
        out = self.fade_in(alpha, downscaled, out)  # Blend resolutions

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)  # Add minibatch standard deviation
        return self.final_block(out).view(out.shape[0], -1)