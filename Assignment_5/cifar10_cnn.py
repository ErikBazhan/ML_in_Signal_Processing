"""Reference CIFAR-10 CNN used by Assignment 5.

The model is intentionally larger than a toy MNIST/Fashion-MNIST classifier so
that model size and CPU latency are visible. It is still small enough to train
or fine-tune on a laptop. The model itself is a plain PyTorch CNN; the
assignment notebook adds quantization wrappers and fusion logic later.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

INPUT_SHAPE = (1, 3, 32, 32)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CifarCNN(nn.Module):
    """A compact but non-trivial plain CIFAR-10 classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)),
                    ("bn1", nn.BatchNorm2d(32)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)),
                    ("bn2", nn.BatchNorm2d(32)),
                    ("relu2", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(kernel_size=2)),
                    ("conv3", nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)),
                    ("bn3", nn.BatchNorm2d(64)),
                    ("relu3", nn.ReLU()),
                    ("conv4", nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)),
                    ("bn4", nn.BatchNorm2d(64)),
                    ("relu4", nn.ReLU()),
                    ("pool2", nn.MaxPool2d(kernel_size=2)),
                    ("conv5", nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)),
                    ("bn5", nn.BatchNorm2d(128)),
                    ("relu5", nn.ReLU()),
                    (
                        "conv6",
                        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    ),
                    ("bn6", nn.BatchNorm2d(128)),
                    ("relu6", nn.ReLU()),
                    ("pool3", nn.MaxPool2d(kernel_size=2)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(128 * 4 * 4, 256)),
                    ("relu7", nn.ReLU()),
                    ("fc2", nn.Linear(256, num_classes)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def make_model(num_classes: int = 10) -> CifarCNN:
    return CifarCNN(num_classes=num_classes)


def cifar10_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def cifar10_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def unnormalize_cifar10(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=image.dtype, device=image.device).view(
        3, 1, 1
    )
    std = torch.tensor(CIFAR10_STD, dtype=image.dtype, device=image.device).view(
        3, 1, 1
    )
    return (image * std + mean).clamp(0.0, 1.0)


def load_reference_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> CifarCNN:
    checkpoint_path = Path(checkpoint_path)
    model = make_model()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
