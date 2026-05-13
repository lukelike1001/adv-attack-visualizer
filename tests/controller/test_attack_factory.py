import pytest

from attacks.base_attack import AdversarialAttack
from attacks.constraints.base_constraint import PerturbationConstraint
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from controller.attack_factory import AttackFactory, UnknownAttackError


class StubConstraint(PerturbationConstraint):
    """Minimal constraint stand-in that performs no real math."""

    def apply_update(self, current_image, gradient, original_image, step_size):
        return current_image


class TestAttackFactory:
    @pytest.fixture
    def factory(self) -> AttackFactory:
        return AttackFactory()

    @pytest.fixture
    def constraint(self) -> StubConstraint:
        return StubConstraint(epsilon=0.03)

    def test_creates_fgsm(self, factory, constraint):
        attack = factory.create_attack("fgsm", model=None, config={}, constraint=constraint)
        assert isinstance(attack, FGSMAttack)

    def test_creates_pgd(self, factory, constraint):
        attack = factory.create_attack(
            "pgd", model=None, config={"num_steps": 5}, constraint=constraint
        )
        assert isinstance(attack, PGDAttack)

    def test_result_is_adversarial_attack(self, factory, constraint):
        attack = factory.create_attack("fgsm", model=None, config={}, constraint=constraint)
        assert isinstance(attack, AdversarialAttack)

    def test_unknown_attack_raises(self, factory, constraint):
        with pytest.raises(UnknownAttackError):
            factory.create_attack("unknown", model=None, config={}, constraint=constraint)

    def test_error_message_contains_attack_name(self, factory, constraint):
        with pytest.raises(UnknownAttackError, match="bad_attack"):
            factory.create_attack("bad_attack", model=None, config={}, constraint=constraint)
