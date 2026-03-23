"""Training entrypoint placeholder."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CNR-SenseNet model.")
    parser.add_argument("--model", default="ed", help="Model name registered under project.models.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise NotImplementedError(f"Training pipeline for model '{args.model}' is not implemented yet.")


if __name__ == "__main__":
    main()
