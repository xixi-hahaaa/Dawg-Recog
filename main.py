# main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# --- import models ---
from models.cnn import VGG11
from models.resnet import get_resnet18
from models.vae import ConvVAE, train as vae_train, test as vae_test
from models.gan import ConvGenerator, ConvDiscriminator

# --- import training utils ---
from utils.train import train_model, test_model
from utils.train_gan import train_gan, test_gan

# ---------------------
# Configuration
# ---------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_classes = 15
epochs = 100
batch_size = 64
lr_cnn = 1e-3
lr_others = 3e-4
latent_size = 256  # for VAE/GAN

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# ---------------------
# Data
# ---------------------
transform_small = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

transform_large = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

train_dataset = ImageFolder("./data/dogs/train", transform=transform_small)
val_dataset   = ImageFolder("./data/dogs/val",   transform=transform_small)
test_dataset  = ImageFolder("./data/dogs/test",  transform=transform_small)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

train_dataset_large = ImageFolder("./data/dogs/train", transform=transform_large)
val_dataset_large   = ImageFolder("./data/dogs/val",   transform=transform_large)
test_dataset_large  = ImageFolder("./data/dogs/test",  transform=transform_large)

train_loader_large = DataLoader(train_dataset_large, batch_size=batch_size, shuffle=True)
val_loader_large   = DataLoader(val_dataset_large,   batch_size=batch_size, shuffle=False)
test_loader_large  = DataLoader(test_dataset_large,  batch_size=batch_size, shuffle=False)


# ---------------------
# Helper function to train & save standard models
# ---------------------
def train_and_save(model, model_name, train_loader, val_loader, epochs, device, lr):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining {model_name}...")

    train_loss, train_acc, val_loss, val_acc = train_model(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}_model.pth"))
    torch.save({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    }, os.path.join(checkpoint_dir, f"{model_name}_history.pt"))

    return model, {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    }

# ---------------------
# Train CNN
# ---------------------
cnn_model = VGG11(num_classes=num_classes)
cnn_model, cnn_history = train_and_save(cnn_model, "cnn", train_loader_large, val_loader_large, epochs, device, lr_cnn)

# ---------------------
# Train ResNet
# ---------------------
resnet_model = get_resnet18(num_classes=num_classes)
resnet_model, resnet_history = train_and_save(resnet_model, "resnet18", train_loader_large, val_loader_large, epochs, device, lr_others)

# ---------------------
# Train VAE
# ---------------------
vae_model = ConvVAE(img_channels=3, latent_dim=256, input_size=(128,128)).to(device)
vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr_others)

vae_train_losses, vae_train_recon, vae_val_losses, vae_val_recon = [], [], [], []

print("\nTraining VAE...")
for epoch in range(epochs):
    print(f"Epoch {epoch+1} out of {epochs}")
    train_loss, train_recon = vae_train(vae_model, vae_optimizer, train_loader, device)
    val_loss, val_recon = vae_test(vae_model, val_loader, device)
    
    vae_train_losses.append(train_loss)
    vae_train_recon.append(train_recon)
    vae_val_losses.append(val_loss)
    vae_val_recon.append(val_recon)

torch.save(vae_model.state_dict(), os.path.join(checkpoint_dir, "vae_model.pth"))
torch.save({
    "vae_train_loss": vae_train_losses,
    "vae_train_recon": vae_train_recon,
    "vae_val_loss": vae_val_losses,
    "vae_val_recon": vae_val_recon
}, os.path.join(checkpoint_dir, "vae_history.pt"))

# ---------------------
# Train GAN
# ---------------------
generator = ConvGenerator(latent_dim=latent_size).to(device)
discriminator = ConvDiscriminator().to(device)
gen_opt = torch.optim.Adam(generator.parameters(), lr=lr_others, betas=(0.5, 0.999))
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=lr_others, betas=(0.5, 0.999))

print("\nTraining GAN...")
for epoch in range(epochs):
    g_loss, d_loss = train_gan(generator, discriminator, train_loader, gen_opt, disc_opt, epoch, device)
    print(f"Epoch {epoch+1} | G_loss: {g_loss:.4f} | D_loss: {d_loss:.4f}")

torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "gan_generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "gan_discriminator.pth"))

# save gan history (losses)
# get the accuracies from the last epoch
g_acc = 100 - (d_loss * 100 / (g_loss + d_loss))
d_acc = 100 - g_acc

torch.save({
    "gan_generator_loss": g_loss,
    "gan_discriminator_loss": d_loss,
    "gan_generator_acc": g_acc,
    "gan_discriminator_acc": d_acc
}, os.path.join(checkpoint_dir, "gan_history.pt"))

# ---------------------
# Done
# ---------------------
print("\nAll models trained and saved. Ready for analysis!")
