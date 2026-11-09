"""Run with: python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from camera import calculate_motion
from config import TrainingConfig
from data import create_data_loaders
from model import ProkrastCNN
from train import train


class PipelineTests(unittest.TestCase):
    def test_flow_distinguishes_stationary_and_translated_texture(self):
        import cv2
        image = np.random.default_rng(42).integers(0, 256, (160, 160), dtype=np.uint8)
        mask, visualization = calculate_motion(image, image)
        self.assertEqual(np.count_nonzero(mask), 0)
        self.assertEqual(visualization.shape, (160, 160, 3))
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 0]]), (160, 160))
        mask, _ = calculate_motion(image, shifted)
        self.assertGreater(np.mean(mask[20:-20, 20:-20] > 0), 0.8)

    def test_training_split_checkpoint_and_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(42)
            for name in ("cup", "phone"):
                folder = root / "data" / name
                folder.mkdir(parents=True)
                for index in range(4):
                    pixels = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(folder / f"{index}.png")
            config = TrainingConfig(
                dataset_dir=root / "data", output_dir=root / "models",
                image_size=16, epochs=1, batch_size=4,
            )
            training, validation, names = create_data_loaders(config)
            repeated_training, repeated_validation, _ = create_data_loaders(config)
            self.assertEqual(names, ["cup", "phone"])
            self.assertEqual(training.dataset.indices, repeated_training.dataset.indices)
            self.assertEqual(validation.dataset.indices, repeated_validation.dataset.indices)
            training_set, validation_set = set(training.dataset.indices), set(validation.dataset.indices)
            self.assertFalse(training_set & validation_set)
            self.assertEqual(training_set | validation_set, set(range(8)))
            targets = training.dataset.dataset.targets
            self.assertEqual({targets[i] for i in training_set}, {0, 1})
            self.assertEqual({targets[i] for i in validation_set}, {0, 1})

            checkpoint_path = train(config)
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            self.assertEqual(checkpoint["class_names"], names)
            network = ProkrastCNN(len(names)).eval()
            network.load_state_dict(checkpoint["model_state_dict"])
            with torch.no_grad():
                logits = network(torch.zeros(2, 3, 16, 16))
            self.assertEqual(tuple(logits.shape), (2, 2))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue((config.output_dir / "training.csv").is_file())
            self.assertTrue((config.output_dir / "training.png").is_file())


if __name__ == "__main__":
    unittest.main()
