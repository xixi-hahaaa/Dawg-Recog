import torch
from torch import nn

import torch
from torch import nn

class ConvVAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=256, input_size=(128, 128)):
        """
        img_channels: number of input channels (e.g., 3 for RGB)
        latent_dim: size of the latent vector
        input_size: tuple (H, W) of input image size
        """
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # -----------------
        # Encoder
        # -----------------
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # H/2 x W/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),            # H/4 x W/4
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),           # H/8 x W/8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # H/16 x W/16
            nn.ReLU()
        )

        # Dynamically compute flattened feature size after convs
        self._conv_out_dim = self._get_conv_output_dim(img_channels, input_size)

        # Latent vectors
        self.fc_mu = nn.Linear(self._conv_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._conv_out_dim, latent_dim)

        # -----------------
        # Decoder
        # -----------------
        self.fc_decode = nn.Linear(latent_dim, self._conv_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # H/16 -> H/8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # H/8 -> H/4
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # H/4 -> H/2
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1), # H/2 -> H
            nn.Sigmoid()  # outputs in [0,1]
        )

    def _get_conv_output_dim(self, channels, input_size):
        # Create a dummy tensor to pass through conv layers to compute flattened size
        with torch.no_grad():
            H, W = input_size
            x = torch.zeros(1, channels, H, W)
            x = self.encoder(x)
            return x.view(1, -1).size(1)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        # reshape to conv feature map
        H, W = self.input_size[0] // 16, self.input_size[1] // 16
        h = h.view(h.size(0), 256, H, W)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss_function(reconstructed_x, x, mu, logvar):
    # Use MSE for RGB images
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence, reconstruction_loss


def train(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    train_reconstruction_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        reconstructed, mu, logvar = model(data)
        loss, reconstruction_loss = vae_loss_function(reconstructed, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_reconstruction_loss += reconstruction_loss.item()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_reconstruction_loss = train_reconstruction_loss / len(train_loader.dataset)
    print(f"Train Loss: {avg_train_loss:.4f} - Reconstruction Loss: {avg_train_reconstruction_loss:.4f}")
    return avg_train_loss, avg_train_reconstruction_loss


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    test_reconstruction_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed, mu, logvar = model(data)
            loss, reconstruction_loss = vae_loss_function(reconstructed, data, mu, logvar)

            test_loss += loss.item()
            test_reconstruction_loss += reconstruction_loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_reconstruction_loss = test_reconstruction_loss / len(test_loader.dataset)
    return avg_test_loss, avg_test_reconstruction_loss
