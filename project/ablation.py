"""Ablation study entrypoint placeholder."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run ablation studies for CNR-SenseNet.")


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("Ablation workflow is not implemented yet.")


if __name__ == "__main__":
    main()
