import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cadastro de Clientes')
        self.setWindowIcon(QIcon('icone.png'))
        self.setGeometry(150, 150, 365, 300) 
        self.Interface()

    def Interface(self):
        self.texto1 = QLabel('Insira no campo abaixo o dado/valor: ', self)
        self.texto1.move(10, 33)
        self.adiciona_cliente = QLineEdit(self)
        self.adiciona_cliente.setFixedWidth(255)
        self.adiciona_cliente.move(10, 50)

        self.lista = QListWidget(self)
        self.lista.move(10, 80)

        botao_adiciona = QPushButton('Adicionar', self)
        botao_adiciona.move(270, 80)
        botao_adiciona.clicked.connect(self.adiciona_elemento)
        botao_selecionar = QPushButton('Selecionar', self)
        botao_selecionar.move(270, 110)
        botao_selecionar.clicked.connect(self.seleciona_elemento)
        botao_remover = QPushButton('Remover', self)
        botao_remover.move(270, 140)
        botao_remover.clicked.connect(self.remove_elemento)
        botao_remover_tudo = QPushButton('Remover Tudo', self)
        botao_remover_tudo.move(270, 170)
        botao_remover_tudo.clicked.connect(self.remove_tudo)

        botao1 = QPushButton('SAIR', self)
        botao1.move(275, 260)
        botao1.clicked.connect(self.confirma_saida)

        self.show()

    def adiciona_elemento(self):
        elemento = self.adiciona_cliente.text()
        self.lista.addItem(elemento)
        self.adiciona_cliente.setText('')

    def seleciona_elemento(self):
        elemento = self.lista.currentItem().text()
        print(elemento)

    def remove_elemento(self):
        indice = self.lista.currentRow()
        self.lista.takeItem(indice)

    def remove_tudo(self):
        self.lista.clear()

    def confirma_saida(self):
        opcoes =  QMessageBox.critical(self,
                                       'ATENÇÃO',
                                       'Deseja mesmo sair?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if opcoes == QMessageBox.StandardButton.Yes:
            sys.exit(qt.exec())
        if opcoes == QMessageBox.StandardButton.Cancel:
            pass

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
