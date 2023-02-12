import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QIcon

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meu Programa')
        self.setWindowIcon(QIcon('icone.png'))
        self.setGeometry(150, 150, 480, 320) 
        self.Interface()

    def Interface(self):
        texto1 = QLabel('Olá Mundo!!!', self) 
        texto1.move(100, 50) 

        botao1 = QPushButton('SAIR', self) 
        botao1.move(100, 200)
        botao1.clicked.connect(self.sair) 

        texto2 = QLabel('Fernando Feltrin', self)
        texto2.move(200, 50)

        self.show()

    def sair(self):
        sys.exit(qt.exec())

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
