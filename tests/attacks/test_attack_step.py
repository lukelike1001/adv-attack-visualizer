import pytest

from attacks.attack_step import AttackStep


class TestAttackStep:
    def _make_step(self, **overrides) -> AttackStep:
        defaults = {
            "step_index": 0,
            "perturbed_image": [0.5, 0.6],
            "gradient": [0.1, 0.2],
            "noise": [0.01, 0.02],
            "metadata": {"epsilon": 0.03},
        }
        return AttackStep(**{**defaults, **overrides})

    def test_fields_are_set_correctly(self):
        step = self._make_step()
        assert step.step_index == 0
        assert step.perturbed_image == [0.5, 0.6]
        assert step.gradient == [0.1, 0.2]
        assert step.noise == [0.01, 0.02]
        assert step.metadata == {"epsilon": 0.03}

    def test_is_immutable(self):
        step = self._make_step()
        with pytest.raises(AttributeError):
            step.step_index = 1

    def test_equal_steps_compare_equal(self):
        assert self._make_step() == self._make_step()

    def test_unequal_steps_differ(self):
        assert self._make_step(step_index=0) != self._make_step(step_index=1)
