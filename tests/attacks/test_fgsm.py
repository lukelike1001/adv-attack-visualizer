import numpy as np
import pytest

from attacks.attack_step import AttackStep
from attacks.base_attack import AdversarialAttack
from attacks.fgsm import FGSMAttack
from tests.attacks.helpers import NumpyLinfConstraint


class FixedGradientFGSM(FGSMAttack):
    """FGSMAttack with a fixed uniform gradient, for testing the pipeline."""

    def _compute_gradient(self, image: np.ndarray, target) -> np.ndarray:
        return np.ones_like(image) * 0.5


class TestFGSMAttack:
    @pytest.fixture
    def image(self) -> np.ndarray:
        return np.zeros((3, 32, 32))

    @pytest.fixture
    def attack(self) -> FixedGradientFGSM:
        constraint = NumpyLinfConstraint(epsilon=0.03)
        return FixedGradientFGSM(
            model=None,
            config={"step_size": 0.03},
            constraint=constraint,
        )

    def test_generate_returns_one_step(self, attack, image):
        steps = attack.generate(image)
        assert len(steps) == 1

    def test_step_index_is_zero(self, attack, image):
        step = attack.generate(image)[0]
        assert step.step_index == 0

    def test_noise_equals_perturbed_minus_original(self, attack, image):
        step = attack.generate(image)[0]
        np.testing.assert_allclose(step.noise, step.perturbed_image - image)

    def test_metadata_keys_present(self, attack, image):
        step = attack.generate(image)[0]
        assert "epsilon" in step.metadata
        assert "step_size" in step.metadata

    def test_returns_attack_step_instances(self, attack, image):
        steps = attack.generate(image)
        assert all(isinstance(s, AttackStep) for s in steps)

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            AdversarialAttack(model=None, config={}, constraint=None)

    def test_compute_gradient_raises_not_implemented(self):
        constraint = NumpyLinfConstraint(epsilon=0.03)
        attack = FGSMAttack(model=None, config={}, constraint=constraint)
        with pytest.raises(NotImplementedError):
            attack._compute_gradient(np.zeros((3, 32, 32)), None)
