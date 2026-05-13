from typing import Any, Dict

from attacks.attack_step import AttackStep


class AttackViewModel:
    """
    GUI-facing representation of a single attack step.

    Wraps an AttackStep and its position in the trajectory, exposing only
    what the view needs.  Transformation logic (e.g. tensor-to-array
    conversion, image normalization) belongs here so the view never
    depends on backend data structures directly.

    This class contains no attack logic and no GUI code.
    """

    def __init__(
        self,
        step: AttackStep,
        step_index: int,
        total_steps: int,
    ) -> None:
        """
        Initialize the ViewModel from an AttackStep and its context.

        Args:
            step: The backend AttackStep this view wraps.
            step_index: Zero-based position of this step in the trajectory.
            total_steps: Total number of steps in the trajectory.
        """
        self._step = step
        self._step_index = step_index
        self._total_steps = total_steps

    @property
    def step_index(self) -> int:
        """Zero-based index of this step in the trajectory."""
        return self._step_index

    @property
    def total_steps(self) -> int:
        """Total number of steps in the trajectory."""
        return self._total_steps

    @property
    def image(self) -> Any:
        """Perturbed image at this step."""
        return self._step.perturbed_image

    @property
    def gradient(self) -> Any:
        """Gradient used to produce the perturbation at this step."""
        return self._step.gradient

    @property
    def noise(self) -> Any:
        """Cumulative perturbation from the original image at this step."""
        return self._step.noise

    @property
    def metadata(self) -> Dict[str, Any]:
        """Copy of the step metadata (e.g. epsilon, step size)."""
        return dict(self._step.metadata)
