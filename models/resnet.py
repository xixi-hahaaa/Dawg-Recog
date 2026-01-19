import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_classes, pretrained=True):
    # Load ResNet-18 model, pretrained on ImageNet
    # use weights argument for torchvision >= 0.13
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

    # Replace the final fully connected layer
    # Original: 512 -> 1000 (ImageNet)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model