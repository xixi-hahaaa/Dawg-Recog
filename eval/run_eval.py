# eval/run_eval.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from models.cnn import VGG11
from models.resnet import get_resnet18
from eval.eval import (
    plot_curves,
    evaluate_accuracy,
    confusion_matrix_plot,
    show_misclassified
)

# ---------------------
# Config
# ---------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_classes = 15
batch_size = 64
checkpoint_dir = "../checkpoints"

# ---------------------
# Test Data ONLY
# ---------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225))
])

test_dataset = ImageFolder("../data/dogs/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = test_dataset.classes

# ---------------------
# Load CNN
# ---------------------
cnn = VGG11(num_classes=num_classes).to(device)
cnn.load_state_dict(torch.load(f"{checkpoint_dir}/cnn_model.pth", map_location=device))

acc = evaluate_accuracy(cnn, test_loader, device)
print(f"CNN Test Accuracy: {acc:.2f}%")

confusion_matrix_plot(cnn, test_loader, device, class_names)
show_misclassified(cnn, test_loader, device, class_names)


# ---------------------
# Load ResNet
# ---------------------
resnet = get_resnet18(num_classes=num_classes).to(device)
resnet.load_state_dict(torch.load(f"{checkpoint_dir}/resnet18_model.pth", map_location=device))

resnet_acc = evaluate_accuracy(resnet, test_loader, device)
print(f"ResNet-18 Test Accuracy: {resnet_acc:.2f}%")

confusion_matrix_plot(resnet, test_loader, device, class_names)
show_misclassified(resnet, test_loader, device, class_names)
