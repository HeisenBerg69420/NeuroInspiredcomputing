"""In-memory image classification for the camera and image CLI."""

from pathlib import Path

import numpy as np
from PIL import Image


class CheckpointClassifier:
    """Load the architecture, labels and preprocessing saved by train.py."""

    def __init__(self, model_path):
        import torch
        from model import AlexNet, ProkrastCNN

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        architectures = {"alexnet-v1": AlexNet, "prokrastcnn-v1": ProkrastCNN}
        architecture = checkpoint.get("architecture")
        if architecture not in architectures:
            raise ValueError("Unbekannte Architektur; bitte einen Checkpoint aus train.py verwenden")
        self.labels = checkpoint["class_names"]
        if not self.labels or any(not isinstance(label, str) or not label.strip()
                                  for label in self.labels):
            raise ValueError("Der Checkpoint benötigt nichtleere Klassennamen")
        size = checkpoint["image_size"]
        if not isinstance(size, int) or size < (67 if architecture == "alexnet-v1" else 8):
            raise ValueError("Ungültige Bildgröße im Checkpoint")
        self.size = (size, size)
        self.mean = np.asarray(checkpoint.get("normalization_mean", [0.5] * 3), dtype=np.float32)
        self.std = np.asarray(checkpoint.get("normalization_std", [0.5] * 3), dtype=np.float32)
        if (self.mean.shape != (3,) or self.std.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError("Ungültige Normalisierung im Checkpoint")
        self.network = architectures[architecture](len(self.labels))
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()

    def predict_image(self, image, top_k=1):
        import torch

        if top_k < 1:
            raise ValueError("top-k muss mindestens 1 sein")
        image = image.convert("RGB").resize(self.size, Image.Resampling.BILINEAR)
        pixels = (np.asarray(image, dtype=np.float32) / 255.0 - self.mean) / self.std
        batch = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]))
        with torch.inference_mode():
            logits = self.network(batch)
            if not torch.isfinite(logits).all():
                raise ValueError("Das Modell liefert ungültige Logits (NaN/Inf)")
            scores, indices = logits.softmax(dim=1)[0].topk(min(top_k, len(self.labels)))
        return [(self.labels[index], float(score)) for index, score in zip(indices, scores)]


class OnnxClassifier:
    def __init__(self, model_path, labels_path=None, image_size=None,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), imagenet=False):
        import onnxruntime as ort

        self.imagenet = imagenet
        if imagenet:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        model_path = Path(model_path)
        labels_path = Path(labels_path) if labels_path else model_path.with_suffix(".labels.txt")
        self.labels = labels_path.read_text(encoding="utf-8").splitlines()
        if not self.labels or any(not label.strip() for label in self.labels):
            raise ValueError("Die Label-Datei muss pro Zeile einen nichtleeren Klassennamen enthalten")
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if (self.mean.shape != (3,) or self.std.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError("mean/std müssen je drei endliche Werte enthalten; std muss positiv sein")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Erwartet wird ein Modell mit genau einer Eingabe und einer Logits-Ausgabe")
        self.input = inputs[0]
        if self.input.type != "tensor(float)" or len(self.input.shape) != 4:
            raise ValueError("Das Modell benötigt eine Float32-Eingabe im Format [1, 3, H, W]")
        batch, channels, height, width = self.input.shape
        if isinstance(batch, int) and batch != 1:
            raise ValueError("Das Modell muss einzelne Bilder (Batchgröße 1) unterstützen")
        if isinstance(channels, int) and channels != 3:
            raise ValueError("Das Modell muss drei RGB-Kanäle erwarten")
        if image_size is not None:
            if image_size < 1:
                raise ValueError("image-size muss positiv sein")
            if any(isinstance(size, int) and size != image_size for size in (height, width)):
                raise ValueError("image-size widerspricht der festen Eingabegröße des Modells")
            height = width = image_size
        if not all(isinstance(size, int) and size > 0 for size in (height, width)):
            raise ValueError("Bei dynamischer Bildgröße bitte --image-size angeben")
        self.size = (width, height)
        if imagenet and self.size != (224, 224):
            raise ValueError("Der ImageNet-Modus für AlexNet benötigt 224 x 224 Pixel")
        self.output_name = outputs[0].name

    def predict(self, image_path, top_k=1):
        """Return [(label, softmax_score), ...], sorted by descending score."""
        if top_k < 1:
            raise ValueError("top-k muss mindestens 1 sein")
        with Image.open(image_path) as image:
            return self.predict_image(image, top_k)

    def predict_image(self, image, top_k=1):
        """Classify a PIL image in memory; no temporary file is needed."""
        if top_k < 1:
            raise ValueError("top-k muss mindestens 1 sein")
        image = image.convert("RGB")
        if self.imagenet:
            # Match torchvision AlexNet IMAGENET1K_V1: shorter edge 256, center crop 224.
            width, height = image.size
            if width <= height:
                resized = (256, int(256 * height / width))
            else:
                resized = (int(256 * width / height), 256)
            image = image.resize(resized, Image.Resampling.BILINEAR)
            left = round((resized[0] - 224) / 2)
            top = round((resized[1] - 224) / 2)
            image = image.crop((left, top, left + 224, top + 224))
        else:
            image = image.resize(self.size, Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        batch = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
        logits = np.asarray(self.session.run([self.output_name], {self.input.name: batch})[0])
        if logits.shape != (1, len(self.labels)):
            raise ValueError(
                f"Ausgabeform {logits.shape} passt nicht zu {len(self.labels)} Labels; "
                "erwartet wird [1, Anzahl Klassen]"
            )
        if not np.isfinite(logits).all():
            raise ValueError("Das Modell liefert ungültige Logits (NaN/Inf)")
        scores = np.exp(logits[0].astype(np.float64) - np.max(logits[0]))
        scores /= scores.sum()
        indices = np.argsort(-scores)[:top_k]
        return [(self.labels[index], float(scores[index])) for index in indices]


