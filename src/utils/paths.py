from pathlib import Path


def _find_project_root(marker: str = "README.md") -> Path:
    """
    Walk upward from this file until a directory containing the marker is found.

    Args:
        marker: Filename whose presence identifies the project root.

    Returns:
        Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If no ancestor directory contains the marker.
    """
    current = Path(__file__).resolve().parent
    for directory in [current, *current.parents]:
        if (directory / marker).exists():
            return directory
    raise FileNotFoundError(
        f"Could not locate project root: no ancestor directory contains '{marker}'."
    )


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
