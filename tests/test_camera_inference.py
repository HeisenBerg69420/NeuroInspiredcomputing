import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from camera import classify_boxes
from inference import CheckpointClassifier
from model import ProkrastCNN


class CheckpointInferenceTests(unittest.TestCase):
    def test_checkpoint_metadata_eval_and_camera_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            network = ProkrastCNN(2)
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.zero_()
                network.classifier[-1].bias[1] = 3
            torch.save({
                "architecture": "prokrastcnn-v1",
                "class_names": ["cup", "phone"],
                "image_size": 16,
                "normalization_mean": [0.2, 0.3, 0.4],
                "normalization_std": [0.5, 0.5, 0.5],
                "model_state_dict": network.state_dict(),
            }, path)
            classifier = CheckpointClassifier(path)
            self.assertFalse(classifier.network.training)
            captured = []
            hook = classifier.network.register_forward_pre_hook(
                lambda model, args: captured.append(args[0].clone())
            )
            frame = np.full((20, 30, 3), (0, 0, 255), dtype=np.uint8)
            prediction = classify_boxes(frame, [(0, 0, 30, 20)], classifier)[0]
            hook.remove()
            self.assertEqual(prediction[1], "phone")
            self.assertGreater(prediction[2], 0.95)
            self.assertEqual(tuple(captured[0].shape), (1, 3, 16, 16))
            np.testing.assert_allclose(captured[0][0, :, 0, 0], [1.6, -0.6, -0.8])
            self.assertEqual(classifier.predict_image(Image.fromarray(frame))[0][0], "phone")


if __name__ == "__main__":
    unittest.main()
