# eval/run_eval_gan.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid

from models.gan import ConvGenerator, ConvDiscriminator

# ---------------------
# Config
# ---------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
latent_dim = 256
batch_size = 64
checkpoint_dir = "../checkpoints"

# ---------------------
# Test dataset ONLY
# ---------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

test_dataset = ImageFolder("../data/dogs/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------
# Load models
# ---------------------
generator = ConvGenerator(latent_dim=latent_dim).to(device)
discriminator = ConvDiscriminator().to(device)

generator.load_state_dict(
    torch.load(f"{checkpoint_dir}/gan_generator.pth", map_location=device)
)
discriminator.load_state_dict(
    torch.load(f"{checkpoint_dir}/gan_discriminator.pth", map_location=device)
)

generator.eval()
discriminator.eval()

# ---------------------
# Discriminator accuracy (real vs fake)
# ---------------------
real_correct = 0
fake_correct = 0
total_real = 0
total_fake = 0

with torch.no_grad():
    for real_imgs, _ in test_loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # real images
        real_labels = torch.ones(batch_size, 1).to(device)
        real_preds = discriminator(real_imgs)
        real_correct += (real_preds > 0.5).sum().item()
        total_real += batch_size

        # fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_preds = discriminator(fake_imgs)
        fake_correct += (fake_preds < 0.5).sum().item()
        total_fake += batch_size

real_acc = 100 * real_correct / total_real
fake_acc = 100 * fake_correct / total_fake

print(f"Discriminator accuracy on REAL images: {real_acc:.2f}%")
print(f"Discriminator accuracy on FAKE images: {fake_acc:.2f}%")

# ---------------------
# Visualize generated samples
# ---------------------
def show_generated_samples(generator, n=16):
    z = torch.randn(n, latent_dim).to(device)
    with torch.no_grad():
        samples = generator(z)

    grid = make_grid(samples.cpu(), nrow=4)

    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0))
    plt.title("GAN Generated Samples")
    plt.axis("off")
    plt.show()

show_generated_samples(generator)

# ---------------------
# Optional: discriminator confidence histogram
# ---------------------
def plot_discriminator_confidence(generator, discriminator, loader, n_batches=5):
    real_scores = []
    fake_scores = []

    with torch.no_grad():
        for i, (real_imgs, _) in enumerate(loader):
            if i >= n_batches:
                break

            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_scores.extend(discriminator(real_imgs).cpu().numpy())

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_scores.extend(discriminator(fake_imgs).cpu().numpy())

    plt.hist(real_scores, bins=30, alpha=0.7, label="Real")
    plt.hist(fake_scores, bins=30, alpha=0.7, label="Fake")
    plt.xlabel("Discriminator Output")
    plt.ylabel("Count")
    plt.title("Discriminator Confidence Distribution")
    plt.legend()
    plt.show()

plot_discriminator_confidence(generator, discriminator, test_loader)
