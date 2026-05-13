from typing import Any, Dict, List, Optional

from attacks.attack_step import AttackStep
from attacks.base_attack import AdversarialAttack
from attacks.constraints.base_constraint import PerturbationConstraint


class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Iteratively perturbs an input image by applying constrained gradient
    steps, projecting back into the allowed perturbation region after each
    update.  The full sequence of AttackStep objects is returned, enabling
    time-based trajectory visualization.

    This class is part of the Model layer and contains no GUI logic.
    """

    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        constraint: PerturbationConstraint,
    ) -> None:
        """
        Initialize the PGD attack.

        Args:
            model: The model being attacked.
            config: Configuration parameters. Must contain 'num_steps'.
                    Supports optional 'step_size'; defaults to epsilon.
            constraint: Perturbation constraint (e.g., L∞ or L2) that
                        handles gradient transformation and projection.
        """
        super().__init__(model, config, constraint)
        self._num_steps: int = config["num_steps"]

    def generate(
        self,
        input_image: Any,
        target: Optional[Any] = None,
    ) -> List[AttackStep]:
        """
        Run the PGD attack and return its full step trajectory.

        Args:
            input_image: The original, unperturbed input image.
            target: Optional target label for targeted attacks.

        Returns:
            A list of AttackStep objects, one per iteration.
        """
        current_image = self._initialize(input_image)
        steps = []
        for step_index in range(self._num_steps):
            step = self._step(current_image, input_image, target, step_index)
            steps.append(step)
            current_image = step.perturbed_image
        return steps

    def _compute_gradient(self, _image: Any, _target: Optional[Any]) -> Any:
        """
        Compute the gradient of the loss with respect to the input image.

        Args:
            image: The current input image.
            target: Optional target label.

        Returns:
            Gradient used to determine the perturbation direction.
        """
        raise NotImplementedError(
            "Gradient computation requires a concrete tensor library implementation."
        )
