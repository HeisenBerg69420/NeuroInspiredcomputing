import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera, CNN training and ONNX export")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("camera", help="Run motion detection")
    commands.add_parser("train", help="Train the CNN", add_help=False)
    commands.add_parser("export", help="Export ONNX", add_help=False)
    args, remaining = parser.parse_known_args()

    if args.command == "camera":
        if remaining:
            parser.error(f"Unknown camera arguments: {' '.join(remaining)}")
        from camera import main as run
        run()
    elif args.command == "train":
        from train import main as run
        run(remaining)
    else:
        from export import main as run
        run(remaining)


if __name__ == "__main__":
    main()
