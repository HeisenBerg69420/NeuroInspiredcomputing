"""Classify images with an ONNX CNN: python test.py --help.

Expected model: one float32 NCHW RGB input and one [1, classes] logits output.
Default normalization matches this project's training (mean/std = 0.5).
"""

import argparse
from pathlib import Path

import onnxruntime as ort
from inference import OnnxClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(paths):
    images = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            images.extend(sorted(p for p in path.rglob("*")
                                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.is_file():
            images.append(path)
        else:
            raise ValueError(f"Bild oder Ordner nicht gefunden: {path}")
    if not images:
        raise ValueError("Keine Bilder gefunden")
    return images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Bilder oder Bildordner")
    parser.add_argument("--model", required=True, type=Path, help="ONNX-Datei")
    parser.add_argument("--labels", type=Path, help="Standard: <Modellname>.labels.txt")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--image-size", type=int, help="Bildgröße für dynamische Modelle")
    parser.add_argument("--mean", nargs=3, type=float, default=(0.5, 0.5, 0.5))
    parser.add_argument("--std", nargs=3, type=float, default=(0.5, 0.5, 0.5))
    args = parser.parse_args(argv)
    try:
        images = collect_images(args.images)
        classifier = OnnxClassifier(args.model, args.labels, args.image_size, args.mean, args.std)
        for image in images:
            predictions = classifier.predict(image, args.top_k)
            print(f"{image}: " + ", ".join(f"{label} ({score:.1%})" for label, score in predictions))
    except (ValueError, OSError, ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            ort.capi.onnxruntime_pybind11_state.NoSuchFile,
            ort.capi.onnxruntime_pybind11_state.InvalidGraph) as error:
        parser.exit(1, f"Fehler: {error}\n")


if __name__ == "__main__":
    main()
