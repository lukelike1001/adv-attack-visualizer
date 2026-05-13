from typing import Any, Dict, Type

from attacks.base_attack import AdversarialAttack
from attacks.constraints.base_constraint import PerturbationConstraint
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack


class UnknownAttackError(Exception):
    """Raised when an unrecognized attack name is requested from the factory."""


class AttackFactory:
    """
    Creates AdversarialAttack instances from string identifiers.

    Centralizes construction logic so that callers do not need to import or
    reference concrete attack classes directly.  New attacks can be supported
    by registering them in _ATTACK_REGISTRY.

    Follows the Factory design pattern (DESIGN-19).
    """

    _ATTACK_REGISTRY: Dict[str, Type[AdversarialAttack]] = {
        "fgsm": FGSMAttack,
        "pgd": PGDAttack,
    }

    def create_attack(
        self,
        attack_name: str,
        model: Any,
        config: Dict[str, Any],
        constraint: PerturbationConstraint,
    ) -> AdversarialAttack:
        """
        Instantiate and return the attack corresponding to attack_name.

        Args:
            attack_name: String identifier for the attack (e.g. 'fgsm', 'pgd').
            model: The model to be attacked.
            config: Configuration dictionary for the attack.
            constraint: Perturbation constraint to apply during the attack.

        Returns:
            A fully initialized AdversarialAttack instance.

        Raises:
            UnknownAttackError: If attack_name does not match any registered attack.
        """
        attack_class = self._ATTACK_REGISTRY.get(attack_name)
        if attack_class is None:
            raise UnknownAttackError(
                f"Unknown attack: '{attack_name}'. "
                f"Available attacks: {list(self._ATTACK_REGISTRY.keys())}"
            )
        return attack_class(model, config, constraint)
