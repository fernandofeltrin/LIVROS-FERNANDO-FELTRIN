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
        texto1 = QLabel('Olá Mundo!!!', self) # Será criada uma linha de texto na posição inicial da janela, que por padrão é o canto superior esquerdo
        texto1.resize(250, 250) # Redefine o tamanho só desse bloco de elementos, para que outros não o sobreponham
        texto1.move(100, 50) # Define manualmente a posição onde texto1 será inserida, nesse caso, 100 pixels a partir da esquerda, 50 pixels a partir do topo. Se não definir este parâmetro para os labels, seus respectivos textos aparecerão sobrepostos.

        texto2 = QLabel('Fernando Feltrin', self)
        texto2.resize(250, 250)
        texto2.move(200, 50)

        self.show()

qt = QApplication(sys.argv)
app = JanelaPrincipal()
sys.exit(qt.exec())
