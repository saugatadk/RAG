from pathlib import Path
from typing import Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Optional[str], default_relative: str) -> Path:
    root = project_root()
    if path:
        target = Path(path)
        if not target.is_absolute():
            target = root / target
    else:
        target = root / default_relative
    return target
