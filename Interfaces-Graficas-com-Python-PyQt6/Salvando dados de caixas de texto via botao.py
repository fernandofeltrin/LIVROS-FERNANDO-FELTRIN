import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meu Programa') 
        self.setWindowIcon(QIcon('icone.png'))
        self.setGeometry(150, 150, 480, 320) 
        self.Interface()

    def Interface(self):
        texto1 = QLabel('Login: ', self)
        texto1.move(40, 50) 

        botao1 = QPushButton('SAIR', self) 
        botao1.move(117, 270)
        botao1.clicked.connect(self.sair) 

        self.caixa_texto1 = QLineEdit(self) 
        self.caixa_texto1.setPlaceholderText('Digite seu nome de usuário')
        self.caixa_texto1.move(90, 48) 

        texto2 = QLabel('Senha: ', self)
        texto2.move(40, 74)
        self.caixa_texto2 = QLineEdit(self)
        self.caixa_texto2.setPlaceholderText('Digite sua senha')
        self.caixa_texto2.setEchoMode(QLineEdit.EchoMode.Password) 
        self.caixa_texto2.move(90, 72)

        botao2 = QPushButton('ENTRAR', self)
        botao2.clicked.connect(self.salva_dados)
        botao2.move(117, 98)

        self.show()

    def sair(self):
        sys.exit(qt.exec())

    def salva_dados(self):
        base = []
        base.append(self.caixa_texto1.text()) 
        base.append(self.caixa_texto2.text())
        print(base) 

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
