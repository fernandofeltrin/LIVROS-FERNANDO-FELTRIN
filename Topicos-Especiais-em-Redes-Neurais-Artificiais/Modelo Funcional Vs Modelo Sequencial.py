# Modelo sequencial convencional
model = Sequential()
model.add(Conv2D(32, 3, activation = 'relu',
                 input_shape = (32, 32, 3)))
model.add(Conv2D(32, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
print(model.summary()) 

# Método sequencial funcional
input_img = Input(shape=(32,32,3))  
conv1_1 = Conv2D(32, kernel_size=3, activation='relu')(input_img)
conv1_2 = Conv2D(32, kernel_size=3, activation='relu')(conv1_1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
flat = Flatten()(pool2)
hidden1 = Dense(128, activation='relu')(flat)
output = Dense(10, activation='softmax')(hidden1)
model_func = Model(inputs=input_img, outputs=output)

model_func.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
print(model_func.summary())

########################################################################

# Exemplo de aplicação
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

input1 = Input(shape=(64,64,1))
conv11 = Conv2D(32, kernel_size=4, activation='relu')(input1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)

input2 = Input(shape=(32,32,3))
conv21 = Conv2D(32, kernel_size=4, activation='relu')(input2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(16, kernel_size=4, activation='relu')(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)

merge = concatenate([flat1, flat2])

hidden = Dense(10, activation='relu')(merge)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs = [input1, input2],
              outputs = output)

print(model.summary())
