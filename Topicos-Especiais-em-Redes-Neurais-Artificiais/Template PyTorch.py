import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from datetime import datetime
from torch import optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(123)

transform = transforms.Compose([transforms.Resize([128,128]),
                                transforms.ToTensor()])

data_treino = datasets.ImageFolder(data_treino,
                                   transform = transform)
data_treino_loader = torch.utils.data.DataLoader(data_treino,
                                                 batch_size = 32,
                                                 shuffle = True)

data_teste = datasets.ImageFolder(data_teste,
                                  transform = transform)
data_teste_loader = torch.utils.data.DataLoader(data_teste,
                                                batch_size = 32,
                                                shuffle = True)

classificador = models.vgg16(pretrained=True)
n_inputs = classificador.classifier[6].in_features
classification_layer = nn.Linear(n_inputs, len(data_treino.classes))
classificador.classifier[6] = classification_layer

print(n_inputs, len(data_treino.classes))

for param in classificador.features.parameters():
    param.requires_grad = False # Desliga os ajustes das descidas dos gradientes, "congelando" as camadas carregadas e pré-treinadas

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classificador.parameters())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classificador.to(device)

def training_loop(loader, epoch):
  running_loss = 0.
  running_accuracy = 0.

  for i, data in enumerate(loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = classificador(inputs)
    loss = loss_criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    predicted = torch.argmax(F.softmax(outputs, dim = 1), dim = 1)
    equals = predicted
    accuracy = torch.mean(equals.float())
    running_accuracy += accuracy
    print(f'Epoca: {epoch + 1}, Loop: {i + 1}')
    print('Loss {%.2f}, Accuracy {%.2f}' % (loss, running_accuracy / len(loader)))

for epoch in range(10):
  training_loop(data_treino_loader, epoch)
  classificador.eval()
  training_loop(data_teste_loader, epoch)
  classificador.train()

def classificar_imagem(fname):
  imagem_teste = Image.open(fname)
  plt.imshow(imagem_teste)
  imagem_teste = imagem_teste.resize((128,128))
  imagem_teste = np.array(imagem_teste.getdata()).reshape(*imagem_teste.size, 3)
  imagem_teste = imagem_teste / 255
  imagem_teste = imagem_teste.transpose(2, 0, 1)
  imagem_teste = torch.tensor(imagem_teste, dtype=torch.float).view(-1, *imagem_teste.shape)
  classificador.eval()
  imagem_teste = imagem_teste.to(device)
  output = classificador.forward(imagem_teste)
  if output > 0.5:
    output = 1
  else:
    output = 0
  print('Previsão: ', output)
  idx_to_class = {value: key for key, value in data_teste.class_to_idx.items()}
  return idx_to_class[output]

