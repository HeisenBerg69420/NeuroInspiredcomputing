"""Export a training checkpoint to ONNX and verify it with ONNX Runtime."""

import argparse
from pathlib import Path

import torch

from config import MODELS_DIR
from model import ProkrastCNN


def export_onnx(checkpoint_path: Path, output_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "ProkrastCNN-baseline-v1":
        raise ValueError("Unsupported checkpoint architecture; use a checkpoint from train.py")
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    network = ProkrastCNN(len(class_names))
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        network, (torch.zeros(1, 3, image_size, image_size),),
        f=str(output_path), input_names=["images"], output_names=["logits"],
        dynamo=True, external_data=False, verify=True,
    )
    output_path.with_suffix(".labels.txt").write_text(
        "\n".join(class_names) + "\n", encoding="utf-8"
    )
    print(f"ONNX model: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "PoviNet.pt")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "PoviNet.onnx")
    args = parser.parse_args(argv)
    export_onnx(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
