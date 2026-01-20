import sys
from PyQt5 import QtWidgets
from ui.main_window import PoseAnalysisGUI


def main():
    """Run the application."""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look

    window = PoseAnalysisGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
