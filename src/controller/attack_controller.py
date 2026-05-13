from typing import Any, List, Optional

from attacks.attack_step import AttackStep
from attacks.base_attack import AdversarialAttack


class AttackNotRunError(Exception):
    """Raised when step access is attempted before any attack has been run."""


class AttackController:
    """
    Coordinates attack execution and manages the resulting step trajectory.

    The controller is the single source of truth for attack execution state.
    It delegates all attack logic to AdversarialAttack implementations and
    exposes the resulting AttackStep sequence for consumption by the view.

    This class contains no attack logic and no GUI code.
    """

    def __init__(self) -> None:
        self._steps: List[AttackStep] = []
        self._current_index: int = 0
        self._current_attack: Optional[AdversarialAttack] = None

    def run_attack(
        self,
        attack: AdversarialAttack,
        input_image: Any,
        target: Optional[Any] = None,
    ) -> None:
        """
        Execute an attack and store the resulting step trajectory.

        Args:
            attack: The adversarial attack to run.
            input_image: The original, unperturbed input image.
            target: Optional target label for targeted attacks.
        """
        self._current_attack = attack
        self._steps = attack.generate(input_image, target)
        self._current_index = 0

    def get_current_step(self) -> AttackStep:
        """
        Return the AttackStep at the current index.

        Returns:
            The current AttackStep in the trajectory.

        Raises:
            AttackNotRunError: If no attack has been run yet.
        """
        if not self._steps:
            raise AttackNotRunError(
                "No attack has been run. Call run_attack() before accessing steps."
            )
        return self._steps[self._current_index]

    def get_total_steps(self) -> int:
        """
        Return the total number of steps in the current trajectory.
        """
        return len(self._steps)

    def get_current_index(self) -> int:
        """
        Return the zero-based index of the current step.
        """
        return self._current_index
