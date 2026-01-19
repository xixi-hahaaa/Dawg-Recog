# generation/generate.py

import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

def generate_vae_image(vae_model, latent_dim, device):
    vae_model.eval()

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        img = vae_model.decode(z)
        img = img.squeeze(0).cpu()

    return tensor_to_pil(img)


def generate_gan_image(generator, latent_dim, device):
    generator.eval()

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        img = generator(z)
        img = img.squeeze(0).cpu()

    return tensor_to_pil(img)


def tensor_to_pil(img_tensor):
    """
    img_tensor: (C, H, W) in [0,1]
    """
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)
