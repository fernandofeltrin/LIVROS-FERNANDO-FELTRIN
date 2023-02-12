# Arquivo base.py
base = [('estou muito feliz','emoção positiva'),
        ('sou uma pessoa feliz','emoção positiva'),
        ('alegria é o meu lema','emoção positiva'),
        ('muito bom ser amado','emoção positiva'),
        ('estou empolgado em começar','emoção positiva'),
        ('fui elogiado por meu trabalho','emoção positiva'),
        ('vencemos a partida','emoção positiva'),
        ('recebi uma promoção de cargo','emoção positiva'),
        ('o dia está muito bonito','emoção positiva'),
        ('estou bem, obrigado','emoção positiva'),
        ('fui aprovado','emoção positiva'),
        ('de bem com a vida','emoção positiva'),
        ('fui bem recebido em casa','emoção positiva'),
        ('estou com medo','emoção negativa'),
        ('estou com muito medo','emoção negativa'),
        ('estou um pouco triste','emoção negativa'),
        ('isto me deixou com raiva','emoção negativa'),
        ('fui demitida','emoção negativa'),
        ('esta comida está horrível','emoção negativa'),
        ('tenho pavor disso','emoção negativa'),
        ('estou incomodado','emoção negativa'),
        ('fiquei desmotivada com o resultado','emoção negativa'),
        ('fui reprovado','emoção negativa')]



# Arquivo mineracao.py
import nltk
from base import base

nltk.download('popular')

stopwords = nltk.corpus.stopwords.words('portuguese')

def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        removesw = [p for p in palavras.split() if p not in stopwords]
        frases.append((removesw, emocao))
    return frases

def reduzpalavras(texto):
    steemer = nltk.stem.RSLPSteemer() # faz o radical das palavras
    frases_redux = []
    for (palavras, emocao) in texto:
        reduzidas = [str(steemer.stem(p)) for p in palavras.split() if p not in stopwords]
        frases_redux.append((reduzidas, emocao))
    return frases_redux

frases_reduzidas = reduzpalavras(base)

def buscapalavras(frases):
    palavras = []
    for (palavras, emocao) in frases:
        palavras.extend(palavras)
    return palavras

palavras = buscapalavras(frases_reduzidas)

def buscafrequencia(palavras):
    freq_palavras = nltk.FreqDist(palavras)
    return freq_palavras

frequencia = buscafrequencia(palavras)

def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicas = buscapalavrasunicas(frequencia)

def extrator(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavrasunicas:
        caracteristicas['%s' % palavra] = (palavra in doc)
    return caracteristicas

#caracteristicasfrases = extrator(['tim','gole','nov'])
#testando se algumas determinadas palavras estão na base de dados

baseprocessada = nltk.classify.apply_features(extrator, frases_reduzidas)

print(baseprocessada[0])


classificador = nltk.NaiveBayesClassifier.train(baseprocessada)

teste = str(input('Digite como você está se sentindo: '))
teste_redux = []
redux = nltk.stem.RSLPSteemer()
for (palavras_treino) in teste.split():
    reduzida = [p for p in palavras_treino.split()]
    teste_redux.append(str(redux.stem(reduzida[0])))

resultado = extrator(teste_redux)
print(classificador.classify(resultado))
