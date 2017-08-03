import numpy as np

from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K

if (K.image_data_format() == 'channels_first'):
    bn_axis = 1
else:
    bn_axis = 3

def my_conv(input, num_filters, kernel_size_tuple, strides=1, padding='valid', name='name'):
    x = Conv2D(num_filters, kernel_size_tuple, strides=strides, padding=padding, 
                      use_bias=True, kernel_initializer='he_normal', name=name)(input)
    return x

def BasicBlock(input, numFilters, stride, isConvBlock):
    expansion = 1
    x = my_conv(input, numFilters, (3, 3), strides=stride, padding='same')
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)
    x = Activation('relu')(x)

    x = my_conv(x, numFilters, (3, 3), padding='same')
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)

    if isConvBlock:
        shortcut = my_conv(input, expansion * numFilters, (1, 1), strides = stride)
        shortcut = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)
    else:
        shortcut = input

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def PreActBlock(stage, block, input, numFilters, stride, isConvBlock):
    expansion = 1
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011, name=bn_name_base + '2a')(input)
    x = Activation('relu')(x)

    if isConvBlock:
        shortcut = my_conv(x, expansion * numFilters, (1, 1), strides = stride, name=conv_name_base + '1')
    else:
        shortcut = x
    
    x = my_conv(x, numFilters, (3, 3), strides=stride, padding='same', name=conv_name_base + '2a')

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = my_conv(x, numFilters, (3, 3), padding = 'same', name=conv_name_base + '2b')

    x = layers.add([x, shortcut])

    return x

def BottleneckBlock(input, numFilters, stride, isConvBlock):
    expansion = 4
    x = my_conv(input, numFilters, (1, 1), strides=stride)
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)
    x = Activation('relu')(x)

    x = my_conv(x, numFilters, (3, 3), padding='same')
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)
    x = Activation('relu')(x)

    x = my_conv(x, 4 * numFilters, (1, 1))
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)

    if isConvBlock:
        shortcut = my_conv(input, expansion * numFilters, (1, 1), strides=stride)
        shortcut = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(shortcut)
    else:
        shortcut = input

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x
    

def PreActBottleneck(input, numFilters, stride, isConvBlock):
    expansion = 4
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(input)
    x = Activation('relu')(x)

    if isConvBlock:
        shortcut = my_conv(x, expansion * numFilters, (1, 1), strides = stride)
    else:
        shortcut = x

    x = my_conv(x, numFilters, (1, 1))

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)
    x = Activation('relu')(x)
    x = my_conv(x, numFilters, (3, 3), padding='same')

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.000011)(x)
    x = Activation('relu')(x)
    x = my_conv(x, 4 * numFilters, (1, 1))

    # did not finish...


def make_layer(stage, block, input, numFilters, numBlocks, stride):
    if stride == 1:
        x = block(stage, 'a', input, numFilters, stride, False)
    else:
        x = block(stage, 'a', input, numFilters, stride, True)

    for i in range(numBlocks - 1):
        x = block(stage, chr(ord('b') + i), x, numFilters, 1, False)
    return x

def ResNet_builder(block, num_blocks, input_shape, num_classes):
    img_input = Input(shape=input_shape)
    x = my_conv(img_input, 64, (3, 3), padding='same', name='conv1')
    x = BatchNormalization(axis = bn_axis, momentum=0.1, epsilon=0.000011, name='bn1')(x)
    x = Activation('relu')(x)
    
    x = make_layer(1, block, x, 64, num_blocks[0], 1)
    x = make_layer(2, block, x, 128, num_blocks[1], 2)
    x = make_layer(3, block, x, 256, num_blocks[2], 2)
    x = make_layer(4, block, x, 512, num_blocks[3], 2)
    
    x = AveragePooling2D((4, 4), strides=4)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name = 'dense')(x)
    
    # return Model(inputs=img_input, outputs=out)

    return Model(inputs=img_input, outputs=x)

# def ResNet18_Basic(input_shape, num_classes):
#     return ResNet_builder(BasicBlock, [2, 2, 2, 2], input_shape, num_classes)
    
def ResNet18(input_shape, num_classes):
    return ResNet_builder(PreActBlock, [2, 2, 2, 2], input_shape, num_classes)

# def ResNet34(input_shape, num_classes):
#     return ResNet_builder(BasicBlock, [3, 4, 6 ,3], input_shape, num_classes)

# def ResNet50(input_shape, num_classes):
#     return ResNet_builder(BottleneckBlock, [3, 4, 6, 3], input_shape, num_classes)

# def ResNet101(input_shape, num_classes):
#     return ResNet_builder(BottleneckBlock, [3, 4, 23, 3], input_shape, num_classes)
        