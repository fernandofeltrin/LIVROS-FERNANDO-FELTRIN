import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
from PyQt6.QtGui import *

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meu Programa')
        self.setWindowIcon(QIcon('icone.png'))
        self.setGeometry(150, 150, 350, 300)
        self.Interface()

    def Interface(self):
        layout2 = QVBoxLayout()

        botao1 = QPushButton('Botão 1', self)
        botao2 = QPushButton('Botão 2', self)
        botao3 = QPushButton('Botão 3', self)

        layout2.setAlignment(Qt.AlignmentFlag.)
        layout2.addStretch()
        layout2.addWidget(botao1)
        layout2.addWidget(botao2)
        layout2.addWidget(botao3)
        layout2.addStretch()

        self.setLayout(layout2)
        self.show()

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
