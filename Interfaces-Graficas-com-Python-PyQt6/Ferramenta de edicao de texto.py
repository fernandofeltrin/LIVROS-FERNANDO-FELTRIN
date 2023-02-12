import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

fonte = QFont('Times', 12) 

class JanelaPrincipal(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Meu Programa') 
        self.setWindowIcon(QIcon('icone.png'))
        self.setGeometry(150, 150, 300, 278) 
        self.Interface()

    def Interface(self):
        texto1 = QLabel('Login: ', self) 
        texto1.move(40, 10) 

        botao1 = QPushButton('SAIR', self) 
        botao1.move(117, 220)
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
        botao2.clicked.connect(self.envia_argumento) 
        botao2.move(117, 114)

        botao_sobre = QPushButton('Sobre', self)
        botao_sobre.move(220, 250)
        botao_sobre.clicked.connect(self.sobre)

        # Caixa de controle de valor
        self.tamanho_fonte = QSpinBox(self) 
        self.tamanho_fonte.move(107, 250)
        texto_tamanho_fonte = QLabel('Tamanho da Fonte:', self)
        texto_tamanho_fonte.move(5, 253)
        self.tamanho_fonte.setMinimum(100) 
        self.tamanho_fonte.setMaximum(150)
        # pode ser reduzido por self.tamanho_fonte.setRange(1, 10)
        self.tamanho_fonte.setSuffix('%') 
        self.tamanho_fonte.setSingleStep(10)
        self.tamanho_fonte.valueChanged.connect(self.tam_fonte)

        self.texto = QTextEdit(self) 
        self.texto.setFixedWidth(280)
        self.texto.setFixedHeight(40)
        self.texto.move(10, 158)
        argumentos = QLabel('Insira suas observações abaixo:', self)
        argumentos.move(10, 143)
        self.texto_checkbox = QCheckBox('Salvar observações', self)
        self.texto_checkbox.move(10, 198)

        self.show()

    def sair(self):
        sys.exit(qt.exec_()) 

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

    def envia_argumento(self):
        if self.texto_checkbox.isChecked(): 
            observacoes = []
            observacoes.append(self.texto.toPlainText()) 
            print('Observações salvas com sucesso.')
            print(f'{observacoes}')
        else:
            print('O usuário não inseriu nenhuma observação.')

    def sobre(self):
        sobre = QMessageBox.information(self, 'Meu Programa', 'Versão 1.0.2')

    def tam_fonte(self):
        valor = self.tamanho_fonte.value()
        fonte0 = QFont('Times', 10)
        fonte1 = QFont('Times', 11)
        fonte2 = QFont('Times', 12)
        fonte3 = QFont('Times', 13)
        fonte4 = QFont('Times', 14)
        fonte5 = QFont('Times', 15)
        if valor == 110:
            self.caixa_texto1.setFont(fonte1)
            self.caixa_texto2.setFont(fonte1)
        elif valor == 120:
            self.caixa_texto1.setFont(fonte2)
            self.caixa_texto2.setFont(fonte2)
        elif valor == 130:
            self.caixa_texto1.setFont(fonte3)
            self.caixa_texto2.setFont(fonte3)
        elif valor == 140:
            self.caixa_texto1.setFont(fonte4)
            self.caixa_texto2.setFont(fonte4)
        elif valor == 150:
            self.caixa_texto1.setFont(fonte5)
            self.caixa_texto2.setFont(fonte5)
        else:
            self.caixa_texto1.setFont(fonte0)
            self.caixa_texto2.setFont(fonte0)

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
