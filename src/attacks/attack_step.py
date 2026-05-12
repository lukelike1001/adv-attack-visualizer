from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class AttackStep:
    """
    Immutable data container representing a single step in an adversarial attack.

    This class is part of the model layer and is intentionally kept free of any
    computation or GUI-related logic. It captures the state of an attack at a
    specific iteration, enabling time-based visualization and analysis.
    """

    step_index: int
    perturbed_image: Any
    gradient: Any
    noise: Any
    metadata: Dict[str, Any]