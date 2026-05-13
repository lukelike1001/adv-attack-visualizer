from pathlib import Path

import pytest

from utils.config_loader import AttackConfigNotFoundError, get_attack_config, load_config

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "attack_config.yaml"


class TestLoadConfig:
    def test_loads_yaml_file(self):
        config = load_config(str(CONFIG_PATH))
        assert isinstance(config, dict)

    def test_contains_expected_attacks(self):
        config = load_config(str(CONFIG_PATH))
        assert "fgsm" in config
        assert "pgd" in config

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestGetAttackConfig:
    @pytest.fixture
    def config(self):
        return load_config(str(CONFIG_PATH))

    def test_returns_fgsm_config(self, config):
        attack_config = get_attack_config(config, "fgsm")
        assert "epsilon" in attack_config
        assert "step_size" in attack_config

    def test_returns_pgd_config(self, config):
        attack_config = get_attack_config(config, "pgd")
        assert "epsilon" in attack_config
        assert "step_size" in attack_config
        assert "num_steps" in attack_config

    def test_unknown_attack_raises(self, config):
        with pytest.raises(AttackConfigNotFoundError):
            get_attack_config(config, "unknown_attack")

    def test_error_message_lists_available_attacks(self, config):
        with pytest.raises(AttackConfigNotFoundError, match="fgsm"):
            get_attack_config(config, "nonexistent")

    def test_fgsm_epsilon_is_positive(self, config):
        attack_config = get_attack_config(config, "fgsm")
        assert attack_config["epsilon"] > 0

    def test_pgd_num_steps_is_positive_integer(self, config):
        attack_config = get_attack_config(config, "pgd")
        assert isinstance(attack_config["num_steps"], int)
        assert attack_config["num_steps"] > 0
