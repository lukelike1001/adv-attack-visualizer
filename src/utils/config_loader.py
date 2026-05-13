from typing import Any, Dict

import yaml


class AttackConfigNotFoundError(Exception):
    """Raised when a requested attack name has no entry in the configuration."""


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        Dictionary representation of the configuration file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(file_path, "r") as config_file:
        return yaml.safe_load(config_file)


def get_attack_config(config: Dict[str, Any], attack_name: str) -> Dict[str, Any]:
    """
    Retrieve the configuration block for a specific attack.

    Args:
        config: Full configuration dictionary, as returned by load_config.
        attack_name: Name of the attack (e.g. 'fgsm', 'pgd').

    Returns:
        Configuration dictionary for the requested attack.

    Raises:
        AttackConfigNotFoundError: If no entry exists for the given attack name.
    """
    if attack_name not in config:
        raise AttackConfigNotFoundError(
            f"No configuration found for attack: '{attack_name}'. "
            f"Available attacks: {list(config.keys())}"
        )
    return config[attack_name]
