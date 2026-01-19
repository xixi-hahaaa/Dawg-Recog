# eval.py

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from models.vae import vae_loss_function, train, test
from utils.train import train_model, test_model
from utils.train_gan import train_gan, test_gan


def plot_curves(train_loss, train_acc, test_loss, test_acc):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_acc, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Test Accuracy')
    plt.legend()
    plt.show()
    plt.close()

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def evaluate_vae(model, data_loader, device, loss_fn):
    model.eval()
    total_loss = 0
    batches = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, _, _ = loss_fn(recon, x, mu, logvar)
            total_loss += loss.item()
            batches += 1

    return total_loss / batches

def show_vae_reconstructions(model, data_loader, device, num_images=5):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)

    with torch.no_grad():
        recon, _, _ = model(x)

    x = x.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(15,4))
    for i in range(num_images):
        axes[0,i].imshow(x[i].permute(1,2,0))
        axes[0,i].set_title("Original")
        axes[0,i].axis("off")

        axes[1,i].imshow(recon[i].permute(1,2,0))
        axes[1,i].set_title("Reconstruction")
        axes[1,i].axis("off")

    plt.show()

def show_gan_samples(generator, device, latent_dim, num_images=16):
    generator.eval()
    z = torch.randn(num_images, latent_dim).to(device)

    with torch.no_grad():
        samples = generator(z)

    samples = samples.cpu()

    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].permute(1,2,0))
        ax.axis("off")
    plt.show()



def confusion_matrix_plot(model, data_loader, device, class_names):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(12,12))
    disp.plot(xticks_rotation=90)
    plt.title("Confusion Matrix")
    plt.show()

def show_misclassified(model, data_loader, device, class_names, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), num_images=5):
    misclassified = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for img, pred, label in zip(inputs, preds, labels):
                if pred != label:
                    misclassified.append((img.cpu(), pred.cpu(), label.cpu()))
            if len(misclassified) >= num_images:
                break

    fig, axes = plt.subplots(1,num_images, figsize=(15,5))
    for i, (img, pred, label) in enumerate(misclassified[:num_images]):
        img = img * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
        img = img.permute(1,2,0).numpy()
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(f"P: {class_names[pred]}\nT: {class_names[label]}")
        axes[i].axis('off')
    plt.show()
