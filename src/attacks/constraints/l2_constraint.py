from typing import Any

from attacks.constraints.base_constraint import PerturbationConstraint


class L2Constraint(PerturbationConstraint):
    """
    L2 constraint: limits overall perturbation magnitude.
    """

    def apply_update(
        self,
        current_image: Any,
        gradient: Any,
        original_image: Any,
        step_size: float
    ) -> Any:
        normalized_grad = self._normalize(gradient)
        update = normalized_grad * step_size

        perturbed = current_image + update
        delta = perturbed - original_image

        delta = self._project_to_l2_ball(delta)

        return original_image + delta

    def _normalize(self, gradient: Any) -> Any:
        raise NotImplementedError("Normalization must be implemented with chosen tensor library")

    def _project_to_l2_ball(self, delta: Any) -> Any:
        raise NotImplementedError("Projection must be implemented with chosen tensor library")