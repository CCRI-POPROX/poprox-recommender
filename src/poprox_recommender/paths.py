from pathlib import Path


def src_dir() -> Path:
    return Path(__file__).parent.parent
