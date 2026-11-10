import csv
import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from config import MODELS_DIR
from model import AlexNet, ProkrastCNN, TrainingConfig, create_hierarchical_data_loaders


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def show_predictions(
    model: nn.Module,
    validation_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    number_of_images: int = 20,
) -> None:
    if len(validation_loader.dataset) == 0:
        return
    random_indices = torch.randperm(
        len(validation_loader.dataset)
    )[:number_of_images]
    model.eval()
    selected_images = []
    selected_labels = []

    for index in random_indices:
        image, label = validation_loader.dataset[index.item()]
        selected_images.append(image)
        selected_labels.append(label)

    images = torch.stack(selected_images)
    labels = torch.as_tensor(selected_labels, dtype=torch.long)

    with torch.no_grad():
        outputs = model(images.to(device, non_blocking=True))
        predictions = outputs.argmax(dim=1).cpu()

    display_images = images * 0.5 + 0.5
    figure, axes = plt.subplots(1, len(display_images), figsize=(12, 3))
    axes = np.atleast_1d(axes)

    for axis, image, label, prediction in zip(
        axes,
        display_images,
        labels,
        predictions,
    ):
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        axis.set_title(
            f"Echt: {class_names[label.item()]}\n"
            f"Modell: {class_names[prediction.item()]}"
        )
        axis.axis("off")

    figure.tight_layout()
    plt.show()



def train(config=None, architecture="alexnet", show_plots=True) -> Path:
    config = config or TrainingConfig()
    architectures = {"alexnet": AlexNet, "prokrastcnn": ProkrastCNN}
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    if architecture == "alexnet" and config.image_size < 67:
        raise ValueError("AlexNet requires image_size >= 67")
    torch.manual_seed(config.random_seed)

    # training_loader, validation_loader, class_names = create_data_loaders(config) decise dependent on what dataseet 
    training_loader, validation_loader, class_names = create_hierarchical_data_loaders(config)
    if not len(training_loader) or not len(validation_loader):
        raise ValueError("Training and validation must both contain images")
    device = select_device()
    model = architectures[architecture](number_of_classes=len(class_names)).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print(f"Geraet: {device}")
    print(f"Klassen: {class_names}")
    print(f"Trainings-Batches: {len(training_loader)}")
    print(f"Validation-Batches: {len(validation_loader)}")
    print(model)
    print(f"Loss: {loss_function.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    training_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    output_dir = getattr(config, "output_dir", MODELS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "AlexNet" if architecture == "alexnet" else "PoviNet"
    model_path = output_dir / f"{model_name}.pt"
    history = []
    training_started_at = time.perf_counter()
    iteration = 0
    print(f"iteration {iteration}")

    for epoch in range(config.epochs):
        epoch_started_at = time.perf_counter()
        model.train()
        running_training_loss = 0.0
        correct_training_predictions = 0
        training_samples = 0
        print(f"iteration {iteration}")

        for inputs, labels in training_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            print(f"iteration {iteration}")

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_training_loss += loss.item() * labels.size(0)
            predicted_classes = outputs.argmax(dim=1)
            correct_training_predictions += (
                predicted_classes == labels
            ).sum().item()
            training_samples += labels.size(0)
            iteration += 1
            if iteration % 100 == 0:
                print(f"iteration {iteration}")

        average_training_loss = running_training_loss / training_samples
        training_accuracy = correct_training_predictions / training_samples
        training_losses.append(average_training_loss)

        model.eval()
        running_validation_loss = 0.0
        correct_predictions = 0
        validation_samples = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                running_validation_loss += loss.item() * labels.size(0)

                predicted_classes = outputs.argmax(dim=1)
                correct_predictions += (
                    predicted_classes == labels
                ).sum().item()
                validation_samples += labels.size(0)

        average_validation_loss = (
            running_validation_loss / validation_samples
        )
        validation_losses.append(average_validation_loss)

        validation_accuracy = correct_predictions / validation_samples
        history.append((epoch + 1, average_training_loss, training_accuracy,
                        average_validation_loss, validation_accuracy))
        scheduler.step(average_validation_loss)

        validation_improved = (
            average_validation_loss
            < best_validation_loss - config.early_stopping_min_delta
        )
        if validation_improved:
            best_validation_loss = average_validation_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": config.image_size,
                "architecture": f"{architecture}-v1",
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.5, 0.5, 0.5],
                "epoch": epoch + 1,
            }, model_path)
        else:
            epochs_without_improvement += 1

        epoch_duration = time.perf_counter() - epoch_started_at
        elapsed_duration = time.perf_counter() - training_started_at
        completed_epochs = epoch + 1
        average_epoch_duration = elapsed_duration / completed_epochs
        remaining_epochs = config.epochs - completed_epochs
        estimated_remaining_duration = (
            average_epoch_duration * remaining_epochs
        )

        print(
            f"Epoch {epoch + 1:02d}/{config.epochs} | "
            f"Train Loss: {average_training_loss:.4f} | "
            f"Train Accuracy: {training_accuracy:.2%} | "
            f"Validation Loss: {average_validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Zeit: {format_duration(epoch_duration)} | "
            f"ETA bis Maximum: "
            f"{format_duration(estimated_remaining_duration)}"
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                "Early Stopping: "
                f"{config.early_stopping_patience} Epochen ohne "
                "ausreichende Verbesserung des Validation Loss."
            )
            break

    total_training_duration = time.perf_counter() - training_started_at
    print(f"Gesamte Trainingszeit: {format_duration(total_training_duration)}")
    print(f"Bestes Modell gespeichert: {model_path}")

    epochs = np.arange(1, len(training_losses) + 1)
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training.png")
    if show_plots:
        plt.show()
    plt.close()

    with (output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "training_loss", "training_accuracy",
                         "validation_loss", "validation_accuracy"))
        writer.writerows(history)
    

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if show_plots:
        show_predictions(model, validation_loader, class_names, device)
    return model_path

    
def export_onnx_HierarchicalDatSet() -> None:
    """Compatibility entry point; ONNX export lives in export.py."""
    from export import export_onnx
    export_onnx(MODELS_DIR / "AlexNet.pt", MODELS_DIR / "AlexNet.onnx")


def main(argv=None) -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train AlexNet or ProkrastCNN")
    parser.add_argument("--model", choices=("alexnet", "prokrastcnn"), default="alexnet")
    parser.add_argument("--dataset", type=Path, default=defaults.dataset_dir)
    parser.add_argument("--output", type=Path, default=defaults.output_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    config = replace(
        defaults, dataset_dir=args.dataset, output_dir=args.output,
        epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size,
    )
    if config.epochs < 1 or config.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    train(config, architecture=args.model, show_plots=not args.no_plots)


if __name__ == "__main__":
    main()
