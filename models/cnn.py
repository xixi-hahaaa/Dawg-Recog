# import libaries
import torch

# we want to train vgg11 as a base point
# we note we are using a smaller version on the actual vgg, so we will be implemeting it ourselves***

# define our vgg_11
class VGG11(torch.nn.Module):
    def __init__(self, num_classes=10, input_size=128):
        super(VGG11, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2)
        )

        # compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            dummy = self.features(dummy)
            flatten_size = dummy.numel()  # total number of features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(flatten_size, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    