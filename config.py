"""Shared paths and training defaults; no dependencies on PyTorch or OpenCV."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: Path = PROJECT_DIR / "data"
    output_dir: Path = MODELS_DIR
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    random_seed: int = 42
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self):
        if self.image_size < 4:
            raise ValueError("image_size must be at least 4")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
