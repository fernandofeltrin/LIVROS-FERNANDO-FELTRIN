import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model

def conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, 
               kernel_size = kernel_size, 
               strides=strides, 
               padding = 'same', 
               use_bias = False)(x)
    x = BatchNormalization()(x)
    return x

def sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters, 
                        kernel_size = kernel_size, 
                        strides=strides, 
                        padding = 'same', 
                        use_bias = False)(x)
    x = BatchNormalization()(x)
    return x

def entry_flow(x):
    x = conv_bn(x, filters = 32,
                kernel_size = 3, strides = 2)
    x = ReLU()(x)
    x = conv_bn(x, filters = 64,
                kernel_size = 3, strides = 1)
    tensor = ReLU()(x)
    x = sep_bn(tensor, filters = 128,
               kernel_size = 3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 128,
               kernel_size = 3)
    x = MaxPool2D(pool_size = 3,
                  strides = 2,
                  padding = 'same')(x)
    tensor = conv_bn(tensor, filters = 128,
                     kernel_size = 1, strides = 2)
    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters = 256,
               kernel_size = 3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 256,
               kernel_size = 3)
    x = MaxPool2D(pool_size = 3,
                  strides = 2,
                  padding = 'same')(x)
    tensor = conv_bn(tensor, filters = 256,
                     kernel_size = 1,strides = 2)
    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters = 728,
               kernel_size = 3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 728,
               kernel_size = 3)
    x = MaxPool2D(pool_size = 3,
                  strides = 2,
                  padding = 'same')(x)
    tensor = conv_bn(tensor,
                     filters = 728,
                     kernel_size = 1,
                     strides = 2)
    x = Add()([tensor, x])
    return x

def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
    return tensor

def exit_flow(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, filters = 728,
               kernel_size = 3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 1024,
               kernel_size = 3)
    x = MaxPool2D(pool_size = 3,
                  strides = 2,
                  padding = 'same')(x)
    tensor = conv_bn(tensor, filters = 1024,
                     kernel_size = 1, strides = 2)
    x = Add()([tensor,x])
    x = sep_bn(x, filters = 1536,
               kernel_size = 3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 2048,
               kernel_size = 3)
    x = GlobalAvgPool2D()(x)
    x = Dense (units = 45,
               activation = 'sigmoid')(x)
    return x

input = Input(shape = (299,299,3))
x = entry_flow(input)
x = middle_flow(x)
output = exit_flow(x)

model = Model (inputs = input,
               outputs = output)

model.summary()
