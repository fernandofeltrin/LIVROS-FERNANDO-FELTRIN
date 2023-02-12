import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meu Programa')
        self.setWindowIcon(QIcon('icone.png'))
        self.setGeometry(150, 150, 350, 300)
        self.Interface()

    def Interface(self):
        layout = QFormLayout()

        texto_usuario = QLabel('Usuário: ')
        input_usuario = QLineEdit()
        input_usuario.setPlaceholderText('Digite seu nome de usuário')
        texto_senha = QLabel('Senha: ')
        input_senha = QLineEdit()
        input_senha.setPlaceholderText('Digite sua senha')
        input_senha.setEchoMode(QLineEdit.EchoMode.Password)
        seleciona_ambiente = QComboBox()
        seleciona_ambiente.addItem('Ambiente Comum')
        seleciona_ambiente.addItem('Painel de Controle')

        layout.addRow(texto_usuario, input_usuario) 
        layout.addRow(texto_senha, input_senha) 
        layout.addRow(QLabel('Salvar Informações: '), seleciona_ambiente)

        layout_secundario = QHBoxLayout()
        layout_secundario.addStretch()
        layout_secundario.addWidget(QPushButton('ENTRAR'))
        layout_secundario.addWidget(QPushButton('SAIR'))
        layout_secundario.addStretch()
        layout.addRow(layout_secundario)

        self.setLayout(layout)
        self.show()

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
