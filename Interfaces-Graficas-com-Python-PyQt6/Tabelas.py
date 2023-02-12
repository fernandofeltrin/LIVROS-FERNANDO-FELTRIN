import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meu Programa') 
        self.setGeometry(150, 150, 500, 500) 
        self.Interface()

    def Interface(self):
        layout = QVBoxLayout()

        self.tabela = QTableWidget() 
        self.tabela.setRowCount(10) 
        self.tabela.setColumnCount(4) 
        self.tabela.setHorizontalHeaderItem(0, QTableWidgetItem('Nome'))
        self.tabela.setHorizontalHeaderItem(1, QTableWidgetItem('Idade'))
        self.tabela.setHorizontalHeaderItem(2, QTableWidgetItem('Telefone'))
        self.tabela.setHorizontalHeaderItem(3, QTableWidgetItem('E-mail'))

        botao1 = QPushButton('Salvar')
        botao1.clicked.connect(self.salva_dados)

        self.tabela.setItem(0, 0, QTableWidgetItem('Fernando'))
        self.tabela.setItem(0, 1, QTableWidgetItem('34'))
        self.tabela.setItem(0, 2, QTableWidgetItem('55991357259'))
        self.tabela.setItem(0, 3, QTableWidgetItem('fernando2rad@gmail.com'))
        self.tabela.setItem(2, 0, QTableWidgetItem('Maria'))
        #self.tabela.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        self.tabela.doubleClicked.connect(self.instancia_elemento)

        layout.addWidget(self.tabela)
        layout.addWidget(botao1)

        self.setLayout(layout)
        self.show()

    def instancia_elemento(self):
        for dado in self.tabela.selectedItems():
            print(f'O elemento {dado.text()} está localizado na linha {dado.row()} e na coluna {dado.column()}.')

    def salva_dados(self):
        base = []
        for dado in self.tabela.selectedItems(): 
            base.append(dado)
            print(base) 
            print(f'O elemento {dado.text()} está localizado na linha {dado.row()} e na coluna {dado.column()}.')

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
