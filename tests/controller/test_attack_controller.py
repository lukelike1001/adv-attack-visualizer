import pytest

from attacks.attack_step import AttackStep
from controller.attack_controller import (
    AttackController,
    AttackNotRunError,
    StepNavigationError,
)


def make_step(index: int) -> AttackStep:
    return AttackStep(
        step_index=index,
        perturbed_image=None,
        gradient=None,
        noise=None,
        metadata={},
    )


class StubAttack:
    """Minimal stand-in for AdversarialAttack with a fixed step sequence."""

    def __init__(self, num_steps: int) -> None:
        self._steps = [make_step(i) for i in range(num_steps)]

    def generate(self, image, target=None):
        return self._steps


class TestAttackController:
    @pytest.fixture
    def controller(self) -> AttackController:
        return AttackController()

    @pytest.fixture
    def loaded_controller(self) -> AttackController:
        ctrl = AttackController()
        ctrl.run_attack(StubAttack(3), input_image=None)
        return ctrl

    def test_get_current_step_before_run_raises(self, controller):
        with pytest.raises(AttackNotRunError):
            controller.get_current_step()

    def test_next_step_before_run_raises(self, controller):
        with pytest.raises(AttackNotRunError):
            controller.next_step()

    def test_previous_step_before_run_raises(self, controller):
        with pytest.raises(AttackNotRunError):
            controller.previous_step()

    def test_run_attack_stores_steps(self, controller):
        controller.run_attack(StubAttack(3), input_image=None)
        assert controller.get_total_steps() == 3

    def test_run_attack_resets_index_to_zero(self, controller):
        controller.run_attack(StubAttack(3), input_image=None)
        controller.next_step()
        controller.run_attack(StubAttack(2), input_image=None)
        assert controller.get_current_index() == 0

    def test_get_current_step_returns_first_step(self, loaded_controller):
        step = loaded_controller.get_current_step()
        assert step.step_index == 0

    def test_next_step_advances_index(self, loaded_controller):
        loaded_controller.next_step()
        assert loaded_controller.get_current_index() == 1

    def test_previous_step_retreats_index(self, loaded_controller):
        loaded_controller.next_step()
        loaded_controller.previous_step()
        assert loaded_controller.get_current_index() == 0

    def test_next_step_at_last_raises(self, loaded_controller):
        loaded_controller.next_step()
        loaded_controller.next_step()
        with pytest.raises(StepNavigationError):
            loaded_controller.next_step()

    def test_previous_step_at_first_raises(self, loaded_controller):
        with pytest.raises(StepNavigationError):
            loaded_controller.previous_step()

    def test_has_next_true_mid_trajectory(self, loaded_controller):
        assert loaded_controller.has_next()

    def test_has_next_false_at_last(self, loaded_controller):
        loaded_controller.next_step()
        loaded_controller.next_step()
        assert not loaded_controller.has_next()

    def test_has_previous_false_at_first(self, loaded_controller):
        assert not loaded_controller.has_previous()

    def test_has_previous_true_after_advance(self, loaded_controller):
        loaded_controller.next_step()
        assert loaded_controller.has_previous()

    def test_current_step_reflects_navigation(self, loaded_controller):
        loaded_controller.next_step()
        step = loaded_controller.get_current_step()
        assert step.step_index == 1

    def test_get_total_steps(self, loaded_controller):
        assert loaded_controller.get_total_steps() == 3

    def test_get_current_index_initial(self, loaded_controller):
        assert loaded_controller.get_current_index() == 0
