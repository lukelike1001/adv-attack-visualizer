from typing import Any

from attacks.constraints.base_constraint import PerturbationConstraint


class LinfConstraint(PerturbationConstraint):
    """
    L-infinity constraint: limits maximum per-element perturbation.
    """

    def apply_update(
        self,
        current_image: Any,
        gradient: Any,
        original_image: Any,
        step_size: float
    ) -> Any:
        update = self._sign(gradient) * step_size
        perturbed = current_image + update
        delta = perturbed - original_image
        delta = self._clip(delta, -self._epsilon, self._epsilon)
        return original_image + delta

    def _sign(self, gradient: Any) -> Any:
        raise NotImplementedError("Sign operation must be implemented with chosen tensor library")

    def _clip(self, value: Any, min_val: float, max_val: float) -> Any:
        raise NotImplementedError("Clip operation must be implemented with chosen tensor library")