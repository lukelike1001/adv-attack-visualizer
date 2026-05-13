from typing import Any, Dict, List, Optional

from attacks.attack_step import AttackStep
from attacks.base_attack import AdversarialAttack
from attacks.constraints.base_constraint import PerturbationConstraint


class FGSMAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    Perturbs an input image by one gradient step scaled and bounded by the
    provided PerturbationConstraint.  FGSM always produces exactly one
    AttackStep, making its output compatible with the multi-step trajectory
    interface shared by all AdversarialAttack subclasses.

    This class is part of the Model layer and contains no GUI logic.
    """

    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        constraint: PerturbationConstraint,
    ) -> None:
        """
        Initialize the FGSM attack.

        Args:
            model: The model being attacked.
            config: Configuration parameters. Supports optional 'step_size'
                    key; defaults to epsilon from the constraint.
            constraint: Perturbation constraint (e.g., L∞ or L2) that
                        handles gradient transformation and projection.
        """
        super().__init__(model, config)
        self._constraint = constraint
        self._step_size: float = config.get("step_size", constraint.get_epsilon())

    def generate(
        self,
        input_image: Any,
        target: Optional[Any] = None,
    ) -> List[AttackStep]:
        """
        Run the FGSM attack and return its single-step trajectory.

        Args:
            input_image: The original, unperturbed input image.
            target: Optional target label for targeted attacks.

        Returns:
            A one-element list containing the AttackStep produced.
        """
        current_image = self._initialize(input_image)
        return [self._step(current_image, input_image, target, 0)]

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

    def _step(
        self,
        current_image: Any,
        original_image: Any,
        target: Optional[Any],
        step_index: int,
    ) -> AttackStep:
        """
        Perform a single constrained gradient step.

        Args:
            current_image: The image at the start of this step.
            original_image: The original, unperturbed input image.
            target: Optional target label.
            step_index: Zero-based index of this step in the trajectory.

        Returns:
            AttackStep capturing the result of this step.
        """
        gradient = self._compute_gradient(current_image, target)
        perturbed_image = self._constraint.apply_update(
            current_image, gradient, original_image, self._step_size
        )
        noise = perturbed_image - original_image
        metadata = {
            "epsilon": self._constraint.get_epsilon(),
            "step_size": self._step_size,
        }
        return AttackStep(
            step_index=step_index,
            perturbed_image=perturbed_image,
            gradient=gradient,
            noise=noise,
            metadata=metadata,
        )
