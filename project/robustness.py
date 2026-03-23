"""Robustness experiment entrypoint placeholder."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run robustness experiments for CNR-SenseNet.")


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("Robustness workflow is not implemented yet.")


if __name__ == "__main__":
    main()
