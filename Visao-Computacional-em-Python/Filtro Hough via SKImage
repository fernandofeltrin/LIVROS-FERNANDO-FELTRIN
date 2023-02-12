# Livro VISÃO COMPUTACIONAL EM PYTHON - FERNANDO FELTRIN https://www.amazon.com.br/dp/B08NTW8TNV

from skimage import io
from skimage import filters
from skimage.transform import (hough_line, probabilistic_hough_line)
from skimage.feature import canny

imagem = io.imread('estrada.jpg')
bordas = canny(imagem, 2, 1, 25)
linhas = hough_line(imagem)

detector = probabilistic_hough_line(bordas,
                                    threshold=10,
                                    line_length=5,
                                    line_gap=3)

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
# Livro Visão Computacional em Python - https://www.amazon.com.br/dp/B08NTW8TNV
