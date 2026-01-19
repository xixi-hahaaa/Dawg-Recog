import torch
import torch.nn as nn
import os
from torchvision.utils import save_image


def train_gan(generator, discriminator, train_loader, g_optimizer, d_optimizer, device, epoch, latent_dim=256):
    """
    Trains the GAN for one epoch on the dataset.

    Args:
        generator: ConvGenerator
        discriminator: ConvDiscriminator
        train_loader: DataLoader for training images
        g_optimizer: optimizer for generator
        d_optimizer: optimizer for discriminator
        device: 'cpu' or 'mps'
        latent_dim: size of latent vector

    Returns:
        avg_g_loss, avg_d_loss: scalar averages over dataset
    """
    os.makedirs("samples", exist_ok=True)

    criterion = nn.BCELoss()
    generator.train()
    discriminator.train()

    g_loss_total = 0
    d_loss_total = 0
    for images, _ in train_loader:
        batch_size = images.size(0)
        images = images.to(device)

        # Discriminator step
        d_optimizer.zero_grad()

        # Real images
        real_labels = torch.ones(batch_size, 1).to(device)
        real_output = discriminator(images)
        d_real_loss = criterion(real_output, real_labels)

        # Fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_output = discriminator(fake_images.detach())
        d_fake_loss = criterion(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        d_loss_total += d_loss.item()

        # Generator step
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        save_image(
            fake_images,
            f"samples/epoch_{epoch}.png",
            nrow=4,
            normalize=True
        )
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_labels)  
        g_loss.backward()
        g_optimizer.step()
        g_loss_total += g_loss.item()

    avg_g_loss = g_loss_total / len(train_loader)
    avg_d_loss = d_loss_total / len(train_loader)

    return avg_g_loss, avg_d_loss


@torch.no_grad()
def test_gan(generator, discriminator, test_loader, device, latent_dim=256):
    """
    Evaluate generator and discriminator on test data.
    Returns average generator and discriminator losses.
    """
    criterion = nn.BCELoss()
    generator.eval()
    discriminator.eval()

    g_loss_total = 0
    d_loss_total = 0
    batch_count = 0

    for images, _ in test_loader:
        batch_count += 1
        batch_size = images.size(0)
        images = images.to(device)

        # Discriminator loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_output = discriminator(images)
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_output = discriminator(fake_images)

        d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
        d_loss_total += d_loss.item()

        # Generator loss
        g_loss = criterion(fake_output, real_labels)  
        g_loss_total += g_loss.item()

    avg_g_loss = g_loss_total / batch_count
    avg_d_loss = d_loss_total / batch_count
    return avg_g_loss, avg_d_loss
