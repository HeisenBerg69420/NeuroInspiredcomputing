"""Training, validation and checkpoint creation. Run: python train.py --help."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from config import TrainingConfig
from data import create_data_loaders
from model import ProkrastCNN


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(network, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    network.train(training)
    total_loss, correct, samples = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = network(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += labels.size(0)
    return total_loss / samples, correct / samples


def train(config: TrainingConfig):
    torch.manual_seed(config.random_seed)
    training_loader, validation_loader, class_names = create_data_loaders(config)
    device = select_device()
    network = ProkrastCNN(len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "PoviNet.pt"
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    started_at = time.perf_counter()
    print(f"Device: {device} | Classes: {class_names}")

    for epoch in range(1, config.epochs + 1):
        training_loss, training_accuracy = run_epoch(
            network, training_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            network, validation_loader, loss_function, device
        )
        scheduler.step(validation_loss)
        history.append((epoch, training_loss, training_accuracy,
                        validation_loss, validation_accuracy))
        if validation_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": network.state_dict(),
                "class_names": class_names,
                "image_size": config.image_size,
                "architecture": "ProkrastCNN-baseline-v1",
                "epoch": epoch,
                "validation_loss": validation_loss,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train loss: {training_loss:.4f}, accuracy: {training_accuracy:.2%} | "
            f"Validation loss: {validation_loss:.4f}, accuracy: {validation_accuracy:.2%}"
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            print("Early stopping")
            break

    with (config.output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "training_loss", "training_accuracy",
                         "validation_loss", "validation_accuracy"))
        writer.writerows(history)

    # Save a plot without requiring a desktop window during training.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    axes.plot([row[0] for row in history], [row[1] for row in history], label="Training")
    axes.plot([row[0] for row in history], [row[3] for row in history], label="Validation")
    axes.set(xlabel="Epoch", ylabel="Loss")
    axes.legend()
    figure.tight_layout()
    figure.savefig(config.output_dir / "training.png")
    plt.close(figure)
    print(f"Best checkpoint: {checkpoint_path} | {time.perf_counter() - started_at:.1f}s")
    return checkpoint_path


def main(argv=None):
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=defaults.dataset_dir)
    parser.add_argument("--output", type=Path, default=defaults.output_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    args = parser.parse_args(argv)
    train(TrainingConfig(
        dataset_dir=args.dataset, output_dir=args.output, epochs=args.epochs,
        batch_size=args.batch_size, image_size=args.image_size,
        learning_rate=args.learning_rate,
    ))


if __name__ == "__main__":
    main()
