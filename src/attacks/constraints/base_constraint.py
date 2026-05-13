from abc import ABC, abstractmethod
from typing import Any


class PerturbationConstraint(ABC):
    """
    Abstract base class for perturbation constraints.

    Defines how gradients are transformed into updates and how
    constraints are enforced on perturbed images.
    """

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    @abstractmethod
    def apply_update(
        self,
        current_image: Any,
        gradient: Any,
        original_image: Any,
        step_size: float
    ) -> Any:
        """
        Apply a constrained update step.

        Args:
            current_image: Current perturbed image.
            gradient: Gradient of loss w.r.t. input.
            original_image: Original clean image.
            step_size: Step size for the update.

        Returns:
            Updated image after applying constraint.
        """
        pass

    def get_epsilon(self) -> float:
        return self._epsilon