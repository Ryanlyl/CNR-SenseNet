"""Evaluation entrypoint placeholder."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CNR-SenseNet model.")
    parser.add_argument(
        "--model", default="energy_detector", help="Model name registered under project.models."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise NotImplementedError(f"Evaluation pipeline for model '{args.model}' is not implemented yet.")


if __name__ == "__main__":
    main()
