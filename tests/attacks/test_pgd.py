import numpy as np
import pytest

from attacks.attack_step import AttackStep
from attacks.pgd import PGDAttack
from tests.attacks.helpers import NumpyLinfConstraint

NUM_STEPS = 5
EPSILON = 0.03
STEP_SIZE = 0.01


class FixedGradientPGD(PGDAttack):
    """PGDAttack with a fixed uniform gradient, for testing the pipeline."""

    def _compute_gradient(self, image: np.ndarray, target) -> np.ndarray:
        return np.ones_like(image) * 0.5


class TestPGDAttack:
    @pytest.fixture
    def image(self) -> np.ndarray:
        return np.zeros((3, 32, 32))

    @pytest.fixture
    def attack(self) -> FixedGradientPGD:
        constraint = NumpyLinfConstraint(epsilon=EPSILON)
        return FixedGradientPGD(
            model=None,
            config={"num_steps": NUM_STEPS, "step_size": STEP_SIZE},
            constraint=constraint,
        )

    def test_generate_returns_correct_number_of_steps(self, attack, image):
        steps = attack.generate(image)
        assert len(steps) == NUM_STEPS

    def test_step_indices_are_sequential(self, attack, image):
        steps = attack.generate(image)
        assert [s.step_index for s in steps] == list(range(NUM_STEPS))

    def test_all_steps_are_attack_step_instances(self, attack, image):
        steps = attack.generate(image)
        assert all(isinstance(s, AttackStep) for s in steps)

    def test_perturbation_bounded_by_epsilon(self, attack, image):
        steps = attack.generate(image)
        for step in steps:
            assert np.all(np.abs(step.noise) <= EPSILON + 1e-6)

    def test_missing_num_steps_raises_key_error(self):
        constraint = NumpyLinfConstraint(epsilon=EPSILON)
        with pytest.raises(KeyError):
            FixedGradientPGD(
                model=None,
                config={"step_size": STEP_SIZE},
                constraint=constraint,
            )

    def test_compute_gradient_raises_not_implemented(self):
        constraint = NumpyLinfConstraint(epsilon=EPSILON)
        attack = PGDAttack(
            model=None,
            config={"num_steps": NUM_STEPS},
            constraint=constraint,
        )
        with pytest.raises(NotImplementedError):
            attack._compute_gradient(np.zeros((3, 32, 32)), None)
