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

        botao2 = QPushButton('NOME', self)
        botao2.move(200, 200)
        botao2.clicked.connect(self.nome_maiusculo)

        self.texto2 = QLabel('Fernando Feltrin', self)
        self.texto2.move(200, 50)
        self.texto2.resize(200, 20)

        self.show()

    def sair(self):
        sys.exit(qt.exec())

    def nome_maiusculo(self):
        self.texto2.setText('FERNANDO FELTRIN')

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
