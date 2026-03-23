"""Shared helpers for experiments."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(*parts: str) -> Path:
    """Resolve a path relative to the project package."""
    return PROJECT_ROOT.joinpath(*parts)
