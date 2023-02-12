import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

fonte = QFont('Times', 16)

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meu Programa') 
        self.setWindowIcon(QIcon('icone.png'))
        self.setGeometry(150, 150, 300, 320) 
        self.Interface()

    def Interface(self):
        self.logo = QLabel(self) 
        self.logo.setPixmap(QPixmap('python.png')) 
        self.logo.move(100, 158)

        texto1 = QLabel('Login: ', self) 
        texto1.move(40, 10) 

        botao1 = QPushButton('SAIR', self)
        botao1.move(117, 270)
        botao1.clicked.connect(self.confirma_saida)

        self.caixa_texto1 = QLineEdit(self) 
        self.caixa_texto1.setPlaceholderText('Digite seu nome de usuário')
        self.caixa_texto1.move(90, 8) 

        texto2 = QLabel('Senha: ', self)
        texto2.move(40, 34)
        self.caixa_texto2 = QLineEdit(self)
        self.caixa_texto2.setPlaceholderText('Digite sua senha')
        self.caixa_texto2.setEchoMode(QLineEdit.EchoMode.Password) 
        self.caixa_texto2.move(90, 32)

        self.salvar_checkbox = QCheckBox('Salvar informações', self) 
        self.salvar_checkbox.move(90, 74) 
        self.salvar_checkbox.clicked.connect(self.salva_dados)

        self.seleciona_ambiente = QComboBox(self) 
        self.seleciona_ambiente.move(90, 92) 
        self.seleciona_ambiente.addItems(['Ambiente Comum',
                                          'Painel de Controle']) 

        self.seleciona_tema1 = QRadioButton('Tema Claro', self) 
        self.seleciona_tema1.move(90, 54)
        self.seleciona_tema1.setChecked(True) 
        self.seleciona_tema2 = QRadioButton('Tema Escuro', self) 
        self.seleciona_tema2.move(170, 54)

        botao2 = QPushButton('ENTRAR', self)
        botao2.setFont(fonte) 
        botao2.clicked.connect(self.salva_dados) 
        botao2.clicked.connect(self.sel_ambiente) 
        botao2.clicked.connect(self.sel_tema)
        botao2.move(117, 114)

        self.show()

    def sel_tema(self):
        if self.seleciona_tema1.isChecked(): 
            print(f'Tema Claro escolhido')
        else:
            print(f'Tema Escuro Escolhido')

    def sel_ambiente(self):
        ambiente_selecionado = self.seleciona_ambiente.currentText() 
        print(f'O ambiente escolhido é: {ambiente_selecionado}')

    def salva_dados(self):
        if self.salvar_checkbox.isChecked(): 
            base = []
            base.append(self.caixa_texto1.text()) 
            base.append(self.caixa_texto2.text()) 
            print(f'Nome de usuário: {base[0]} \nSenha: {base[1]}')
        else: 
            print(f'Usuário optou por não salvar os dados.')

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
