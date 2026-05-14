from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt

from controller.attack_controller import AttackController
from utils.config_loader import load_config
from utils.paths import CONFIG_DIR
from view.view_model import AttackViewModel

GUI_CONFIG_PATH = CONFIG_DIR / "gui_config.yaml"


class MainWindow(QMainWindow):
    """
    Main application window for the adversarial attack visualizer.

    Establishes the top-level layout with placeholder regions for the math
    panel, image panel, and controls panel.  Concrete panel implementations
    will replace the placeholders in later GUI issues.

    This class is part of the View layer and contains no attack logic.
    """

    def __init__(self, controller: AttackController) -> None:
        """
        Initialize the main window.

        Args:
            controller: The AttackController that manages attack execution
                        and step navigation.
        """
        super().__init__()
        self._controller = controller
        self._gui_config = load_config(GUI_CONFIG_PATH)
        self._init_window()
        self._init_layout()

    def _init_window(self) -> None:
        window_cfg = self._gui_config["window"]
        self.setWindowTitle(window_cfg["title"])
        self.resize(window_cfg["width"], window_cfg["height"])

    def _init_layout(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)

        panels_cfg = self._gui_config["panels"]
        panel_splitter = QSplitter(Qt.Horizontal)
        panel_splitter.addWidget(self._make_placeholder(panels_cfg["math"]))
        panel_splitter.addWidget(self._make_placeholder(panels_cfg["image"]))

        root_layout.addWidget(panel_splitter)
        root_layout.addWidget(self._make_placeholder(panels_cfg["controls"]))

    def _make_placeholder(self, label: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(label))
        return widget

    def render(self, view_model: AttackViewModel) -> None:
        """
        Update the window to reflect the given view model.

        Args:
            view_model: The AttackViewModel to display.
        """
        pass
