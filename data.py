"""ImageFolder input: data/<class name>/<image>, split reproducibly per class."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import TrainingConfig


def create_data_loaders(config: TrainingConfig):
    if not config.dataset_dir.is_dir():
        raise ValueError(
            f"Dataset not found: {config.dataset_dir}. "
            "Expected one subfolder per class containing images."
        )
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(config.dataset_dir, transform=transform)
    generator = torch.Generator().manual_seed(config.random_seed)
    training_indices, validation_indices = [], []
    for class_index, class_name in enumerate(dataset.classes):
        indices = [i for i, target in enumerate(dataset.targets) if target == class_index]
        if len(indices) < 2:
            raise ValueError(f"Class {class_name!r} needs at least two images")
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in order]
        validation_count = min(
            len(indices) - 1, max(1, round(len(indices) * config.validation_fraction))
        )
        validation_indices.extend(indices[:validation_count])
        training_indices.extend(indices[validation_count:])

    training_loader = DataLoader(
        Subset(dataset, training_indices), batch_size=config.batch_size,
        shuffle=True, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=config.batch_size,
        shuffle=False, num_workers=0,
    )
    return training_loader, validation_loader, dataset.classes
