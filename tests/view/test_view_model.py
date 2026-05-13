import pytest

from attacks.attack_step import AttackStep
from view.view_model import AttackViewModel


class TestAttackViewModel:
    @pytest.fixture
    def step(self) -> AttackStep:
        return AttackStep(
            step_index=2,
            perturbed_image=[0.5, 0.6],
            gradient=[0.1, 0.2],
            noise=[0.01, 0.02],
            metadata={"epsilon": 0.03, "step_size": 0.01},
        )

    @pytest.fixture
    def view_model(self, step) -> AttackViewModel:
        return AttackViewModel(step=step, step_index=2, total_steps=5)

    def test_step_index(self, view_model):
        assert view_model.step_index == 2

    def test_total_steps(self, view_model):
        assert view_model.total_steps == 5

    def test_image(self, view_model):
        assert view_model.image == [0.5, 0.6]

    def test_gradient(self, view_model):
        assert view_model.gradient == [0.1, 0.2]

    def test_noise(self, view_model):
        assert view_model.noise == [0.01, 0.02]

    def test_metadata_values(self, view_model):
        assert view_model.metadata == {"epsilon": 0.03, "step_size": 0.01}

    def test_metadata_is_a_copy(self, view_model):
        meta = view_model.metadata
        meta["epsilon"] = 999
        assert view_model.metadata["epsilon"] == 0.03
