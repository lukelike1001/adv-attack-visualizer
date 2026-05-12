from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from attacks.attack_step import AttackStep


class AdversarialAttack(ABC):
    """
    Abstract base class for all adversarial attacks.

    This class defines the common interface and structure for generating
    adversarial examples. Concrete attack implementations (e.g., FGSM, PGD)
    must subclass this and implement the required methods.

    This class is part of the Model layer and must not depend on any GUI logic.
    """

    def __init__(self, model: Any, config: Dict[str, Any]) -> None:
        """
        Initialize the attack.

        Args:
            model: The model being attacked.
            config: Configuration parameters for the attack.
        """
        self._model = model
        self._config = config

    @abstractmethod
    def generate(
        self,
        input_image: Any,
        target: Optional[Any] = None
    ) -> List[AttackStep]:
        """
        Run the adversarial attack.

        Args:
            input_image: The original input image.
            target: Optional target label for targeted attacks.

        Returns:
            A list of AttackStep objects representing the attack trajectory.
        """
        pass

    @abstractmethod
    def _compute_gradient(self, image: Any, target: Optional[Any]) -> Any:
        """
        Compute the gradient of the loss with respect to the input image.

        Args:
            image: The current input image.
            target: Optional target label.

        Returns:
            Gradient information used to update the image.
        """
        pass

    def _initialize(self, input_image: Any) -> Any:
        """
        Optional initialization step for attacks.

        Args:
            input_image: The original input image.

        Returns:
            Initial state for the attack.
        """
        return input_image

    @abstractmethod
    def _step(self, current_image: Any, target: Optional[Any]) -> Any:
        """
        Perform a single attack step.

        Args:
            current_image: The current perturbed image.
            target: Optional target label.

        Returns:
            Updated image after one attack step.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Return a copy of the attack configuration.
        """
        return dict(self._config)