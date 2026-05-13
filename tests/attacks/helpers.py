import numpy as np

from attacks.attack_step import AttackStep
from attacks.constraints.linf_constraint import LinfConstraint


class NumpyLinfConstraint(LinfConstraint):
    """Concrete LinfConstraint using NumPy, for use in tests only."""

    def _sign(self, gradient: np.ndarray) -> np.ndarray:
        return np.sign(gradient)

    def _clip(self, value: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return np.clip(value, min_val, max_val)


def make_attack_step(index: int = 0) -> AttackStep:
    """Return a minimal AttackStep for use as a test fixture."""
    return AttackStep(
        step_index=index,
        perturbed_image=None,
        gradient=None,
        noise=None,
        metadata={},
    )
