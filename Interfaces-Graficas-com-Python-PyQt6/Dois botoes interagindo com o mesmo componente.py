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
        self.texto1 = QLabel('Olá Mundo!!!', self) 
        self.texto1.move(100, 50) 

        botao1 = QPushButton('Maiusculo', self)
        botao1.move(100, 200)
        botao1.clicked.connect(self.maiusculo)

        botao2 = QPushButton('Invertido', self)
        botao2.move(200, 200)
        botao2.clicked.connect(self.invertido)

        self.show()

    def maiusculo(self):
        self.texto1.setText('OLÁ MUNDO!!!') 
        self.texto1.resize(150, 15) 
    def invertido(self):
        self.texto1.setText('!!!odnuM àlO')

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
