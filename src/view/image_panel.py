from enum import Enum
from typing import Any, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from utils.config_loader import load_config
from utils.paths import CONFIG_DIR
from view.view_model import AttackViewModel

GUI_CONFIG_PATH = CONFIG_DIR / "gui_config.yaml"


class DisplayMode(Enum):
    """Selects which image is rendered in the ImagePanel."""
    ORIGINAL = "original"
    NOISE = "noise"
    PERTURBED = "perturbed"


class ImagePanel(QWidget):
    """
    Displays one of three image views: original, noise map, or perturbed image.

    The panel renders whichever image the current DisplayMode selects.  The
    original image must be supplied separately via set_original_image because
    AttackViewModel only carries the perturbed image and noise; the original
    will be provided by the main window during controller wiring (GUI-05).

    This class is part of the View layer and contains no attack logic.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cfg = load_config(GUI_CONFIG_PATH)["image_panel"]
        self._view_model: Optional[AttackViewModel] = None
        self._original_image: Optional[Any] = None
        self._mode: DisplayMode = DisplayMode.PERTURBED
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._mode_label = QLabel()
        self._image_label = QLabel(self._cfg["placeholder_text"])
        self._image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._mode_label)
        layout.addWidget(self._image_label)

    def set_view_model(self, view_model: AttackViewModel) -> None:
        """
        Update the panel with a new ViewModel and re-render.

        Args:
            view_model: The AttackViewModel to display.
        """
        self._view_model = view_model
        self.render()

    def set_original_image(self, image: Any) -> None:
        """
        Supply the original, unperturbed image for display in ORIGINAL mode.

        Args:
            image: The clean input image passed to the attack.
        """
        self._original_image = image

    def set_display_mode(self, mode: DisplayMode) -> None:
        """
        Switch the active display mode and re-render.

        Args:
            mode: The DisplayMode to activate.
        """
        self._mode = mode
        self.render()

    def render(self) -> None:
        """Redraw the panel based on the current mode and ViewModel."""
        mode_labels = self._cfg["mode_labels"]
        self._mode_label.setText(mode_labels[self._mode.value])

        if self._view_model is None:
            self._image_label.setText(self._cfg["placeholder_text"])
            return

        image_data = self._select_image()
        if image_data is None:
            self._image_label.setText(self._cfg["placeholder_text"])
            return

        pixmap = self._to_pixmap(image_data)
        if pixmap is not None:
            self._image_label.setPixmap(
                pixmap.scaled(
                    self._image_label.width(),
                    self._image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def _select_image(self) -> Optional[Any]:
        if self._mode == DisplayMode.ORIGINAL:
            return self._original_image
        if self._mode == DisplayMode.NOISE:
            return self._view_model.noise
        return self._view_model.image

    def _to_pixmap(self, array: Any) -> Optional[QPixmap]:
        """
        Convert a NumPy array to a QPixmap for Qt display.

        Handles float arrays in [0, 1] and channel-first layout (C, H, W).
        Returns None if conversion is not possible.

        Args:
            array: Image data as a NumPy array.

        Returns:
            QPixmap ready for display, or None if conversion failed.
        """
        try:
            array = np.array(array)

            if array.ndim == 3 and array.shape[0] in (1, 3):
                array = np.transpose(array, (1, 2, 0))

            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]

            if array.dtype != np.uint8:
                array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)

            array = np.ascontiguousarray(array)
            height, width = array.shape[:2]

            if array.ndim == 2:
                q_image = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
            else:
                q_image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)

            return QPixmap.fromImage(q_image)

        except Exception:
            return None
