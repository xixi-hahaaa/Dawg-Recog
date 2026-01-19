# eval/run_eval_vae.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid

from models.vae import ConvVAE, vae_loss_function

# ---------------------
# Config
# ---------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 64
latent_dim = 256
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
# Load VAE
# ---------------------
vae = ConvVAE(
    img_channels=3,
    latent_dim=latent_dim,
    input_size=(128,128)
).to(device)

vae.load_state_dict(
    torch.load(f"{checkpoint_dir}/vae_model.pth", map_location=device)
)

vae.eval()

# ---------------------
# Evaluate test loss
# ---------------------
test_loss = 0
recon_loss = 0

with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        recon, mu, logvar = vae(x)
        loss, recon = vae_loss_function(recon, x, mu, logvar)
        test_loss += loss.item()
        recon_loss += recon.item()

test_loss /= len(test_loader.dataset)
recon_loss /= len(test_loader.dataset)

print(f"VAE Test Loss: {test_loss:.4f}")
print(f"VAE Reconstruction Loss: {recon_loss:.4f}")

# ---------------------
# Visualize reconstructions
# ---------------------
def show_reconstructions(model, loader, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    with torch.no_grad():
        recon, _, _ = model(x)

    comparison = torch.cat([x.cpu(), recon.cpu()])
    grid = make_grid(comparison, nrow=n)

    plt.figure(figsize=(12,4))
    plt.imshow(grid.permute(1,2,0))
    plt.title("Top: Original | Bottom: Reconstruction")
    plt.axis("off")
    plt.show()

show_reconstructions(vae, test_loader)

# ---------------------
# Sample from latent space
# ---------------------
def sample_latent(model, n=16):
    z = torch.randn(n, latent_dim).to(device)
    with torch.no_grad():
        samples = model.decode(z)

    grid = make_grid(samples.cpu(), nrow=4)

    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0))
    plt.title("Random Samples from VAE")
    plt.axis("off")
    plt.show()

sample_latent(vae)

# ---------------------
# Plot saved loss curves (optional)
# ---------------------
try:
    history = torch.load(f"{checkpoint_dir}/vae_history.pt")

    plt.plot(history["vae_train_recon"], label="Train Recon")
    plt.plot(history["vae_test_recon"], label="Test Recon")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("VAE Reconstruction Loss")
    plt.legend()
    plt.show()
except:
    print("No VAE history found — skipping curve plot.")
