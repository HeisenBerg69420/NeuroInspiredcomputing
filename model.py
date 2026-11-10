import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


MODELS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    # dataset_dir: Path = (MODELS_DIR/ "dataset"/ "archive"/ "Human Action Recognition" ) 
    dataset_dir: Path = (MODELS_DIR/ "dataset"/ "Final Dataset 256"/ "Final Dataset 256" ) 
    output_dir: Path = MODELS_DIR / "models"
    image_size: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 3
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    random_seed: int = 42
    validation_fraction: float = 0.15

class ActionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        annotations_path: Path,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.images_dir = images_dir
        self.class_to_index = class_to_index
        self.transform = transform
        self.samples: list[tuple[str, str]] = []

        with annotations_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames != ["filename", "label"]:
                raise ValueError(
                    "Die CSV benötigt die Spalten filename,label"
                )

            for row in reader:
                filename = row["filename"]
                label = row["label"]

                if label not in class_to_index:
                    raise ValueError(f"Unbekanntes Label: {label}")

                image_path = images_dir / filename
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Bild aus CSV nicht gefunden: {image_path}"
                    )

                self.samples.append((filename, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        filename, label = self.samples[index]
        image_path = self.images_dir / filename

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image_tensor = self.transform(image)
        label_index = self.class_to_index[label]

        return image_tensor, torch.tensor(
            label_index,
            dtype=torch.long,
        )


def read_labels(annotations_path: Path) -> list[str]:
    with annotations_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        labels = {row["label"] for row in reader}

    return sorted(labels)

def read_labels_Hierarchical(annotations_path: Path) -> list[str]:
    labels = [f.name for f in annotations_path.iterdir() if f.is_dir()]
    return sorted(labels)

class ProkrastCNN(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)
    
class AlexNet(nn.Module):
    """AlexNet variant with the original 96/256/384/384/256 channel widths."""

    def __init__(self, number_of_classes: int = 1000) -> None:
        super().__init__()
        if number_of_classes < 1:
            raise ValueError("number_of_classes must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, number_of_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(self.avgpool(features))


def create_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalization = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    training_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return training_transforms, evaluation_transforms


def create_hierarchical_data_loaders(
    config: TrainingConfig,
    images_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create loaders for images grouped into class directories.

    Expected layout:

    dataset_dir/
        class_a/
            image_1.jpg
            optional_subdirectory/image_2.jpg
        class_b/
            image_3.jpg
    """
    dataset_root = images_dir or config.dataset_dir
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Datensatz-Ordner nicht gefunden: {dataset_root}"
        )
    if not 0 <= config.validation_fraction < 1:
        raise ValueError(
            "validation_fraction muss zwischen 0 (inklusive) "
            "und 1 (exklusive) liegen"
        )

    training_transforms, evaluation_transforms = create_transforms(
        config.image_size
    )
    training_dataset = datasets.ImageFolder(
        root=dataset_root,
        transform=training_transforms,
    )
    validation_dataset = datasets.ImageFolder(
        root=dataset_root,
        transform=evaluation_transforms,
    )

    if not training_dataset.samples:
        raise ValueError(
            f"Keine unterstützten Bilder gefunden: {dataset_root}"
        )

    generator = torch.Generator().manual_seed(config.random_seed)
    indices = torch.randperm(
        len(training_dataset),
        generator=generator,
    ).tolist()
    validation_size = int(
        len(indices) * config.validation_fraction
    )
    if len(indices) > 1 and config.validation_fraction > 0:
        validation_size = max(1, min(validation_size, len(indices) - 1))

    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    training_loader = DataLoader(
        Subset(training_dataset, training_indices),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return training_loader, validation_loader, training_dataset.classes


