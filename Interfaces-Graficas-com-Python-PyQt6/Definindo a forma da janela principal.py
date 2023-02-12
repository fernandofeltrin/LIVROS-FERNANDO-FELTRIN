import sys
from PyQt6.QtWidgets import *

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(150, 150, 480, 320)

        self.show()

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
