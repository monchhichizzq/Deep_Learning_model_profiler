# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 16:12
# @Author  : Zeqi@@
# @FileName: models.py
# @Software: PyCharm



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Layer,BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling1D, AveragePooling1D
from tensorflow.keras import regularizers
# Not support submodel (get_config() problem)
from Model_design.tensorflow2NNProfiler import NN_profiler, output_nb_param



class Constraint(object):
    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class quantization_constraint_v1(Constraint):
    def __init__(self, max_value):
        '''
        :param max_value: An interger less than 2 power 8
        '''
        self.max_value = max_value

    def __call__(self, w):
        Wmax = K.max(K.abs(w)) + K.epsilon()
        w = K.round(w * self.max_value / Wmax) * Wmax / self.max_value
        return w

class quantization_constraint_v2(Constraint):
    # real_value = (sint8_value — zero_point) * scale
    def __init__(self, max_value):
        '''
        :param max_value: An interger less than 2 power 8
        '''
        self.max_value = max_value

    def __call__(self, w):
        Wmax = K.max(w) - K.min(w) + K.epsilon()
        scale_factor = 2*self.max_value/Wmax
        zero_adjusted_w = w - (K.max(w) + K.min(w))/2
        w = K.round(zero_adjusted_w*scale_factor)/scale_factor
        return w

def conv_block(x,
                  filter_num,
                  kernel_size,
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-5),
                  max_value=2**7-1,
                  activation='relu',
                  pool_size = 1,
                  drop_rate = 0,
                  with_bn = False,
                  *args,
                  **kwargs):

    # Set max_value as 'off' to turn off the kernel_constraint
    if isinstance(max_value, (int, float)):
        x = Conv2D(filter_num,
                   kernel_size,
                   strides = 1,
                   padding = 'same',
                   activation = None,
                   use_bias = False,
                   kernel_initializer = kernel_initializer,
                   kernel_regularizer = kernel_regularizer,
                   kernel_constraint=quantization_constraint_v2(max_value)
                   )(x)
    else:
        x = Conv2D(filter_num,
                   kernel_size,
                   strides=1,
                   padding='same',
                   activation=None,
                   use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer
                   )(x)

    if with_bn:
        x = BatchNormalization(dtype='float32')(x)

    if activation != 0:
        x = Activation(activation, dtype='float32')(x)

    if pool_size != 1:
        x = AveragePooling2D(pool_size=(pool_size, pool_size), padding='same')(x)

    if drop_rate != 0:
        x = Dropout(drop_rate)(x)

    return x

# Example model for IBM dataset
def model_tiny(input_shape,
                num_classes,
                max_value,
                with_bn = False,
                quantize_aware=False,
                mixed_precision=False):
    # main_input = Input(shape=input_shape, dtype='float16')
    main_input = Input(shape=input_shape)
    x = main_input
    if quantize_aware:
        x = BatchNormalization(dtype='float32')(x)

    conv_layers = [8, 16, 32, 32]
    for num_filters in conv_layers:
        x = conv_block(x,
                          num_filters,
                          (3, 3),
                          kernel_regularizer=regularizers.l2(1e-5),
                          max_value=max_value,
                          pool_size=2,
                          drop_rate=0,
                          no_threshold=no_threshold,
                          with_bn=with_bn)

    x = conv_block(x,
                      num_classes,
                      (1, 1),
                      kernel_regularizer=regularizers.l2(1e-5),
                      max_value=max_value,
                      pool_size=1,
                      drop_rate=0,
                      no_threshold=no_threshold,
                      with_bn=with_bn,
                      activation=0)
    gap = GlobalAveragePooling2D()(x)
    predictions = Activation('softmax', dtype='float32')(gap)
    model = Model(inputs=main_input, outputs=predictions)
    return model

def model(input_shape, 
            num_classes, 
            max_value,  
            with_bn = False, 
            input_thr = False, 
            mixed_precision=False, 
            quantize_aware=False):

    main_input = Input(shape=input_shape)
    x = main_input
    if quantize_aware:
        x = BatchNormalization(dtype='float32')(x)

    conv_layers = [16, 32, 32, 64, 64]
    poolings = [2, 2, 2, 1, 2]
    for i, num_filters in enumerate(conv_layers):
        x = conv_block(x,
                          num_filters,
                          (3, 3),
                          kernel_regularizer=regularizers.l2(1e-5),
                          max_value=max_value,
                          pool_size=poolings[i],
                          drop_rate=0,
                          with_bn=with_bn)

    x = conv_block(x,
                      num_classes,
                      (1, 1),
                      kernel_regularizer=regularizers.l2(1e-5),
                      max_value=max_value,
                      pool_size=1,
                      drop_rate=0,
                      with_bn=with_bn,
                      activation=0)

    gap = GlobalAveragePooling2D()(x)
    predictions = Activation('softmax', dtype='float32')(gap)
    model = Model(inputs=main_input, outputs=predictions)
    return model

def T_learning_classifier(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(num_classes, (1,1), strides=1, padding ='same', activation=None, use_bias=False))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape)
    return model

def model_profiler(model):
    model_name = 'sample_model'
    os.makedirs('statistics_files', exist_ok=True)
    print_file_model = open('statistics_files/{0}_report.txt'.format(model_name), 'w+')
    print('\n{0}\n'.format(model_name), file=print_file_model)
    Total_neurons, Total_FlOPS = NN_profiler(model, print_file_model)
    print('\nTotal number of Neurons: {:<10f} K'.format(np.round(Total_neurons / 10 ** 3, 2)), file=print_file_model)
    print('\nTotal number of operations: {:<10f} MFLOPs'.format(np.round(Total_FlOPS / 10 ** 6, 2)),
          file=print_file_model)
    total_param = output_nb_param(model)
    print('\nTotal number of Parameters: {:<10f} K'.format(np.round(total_param / 10 ** 3, 2)), file=print_file_model)
    print_file_model.close()

def detect_kernel_and_feature_map_float_dtype(model):
     for i, layer in enumerate(model.layers):
        layer_class = layer.__class__.__name__
        if layer_class == 'Conv2D':
            print('Conv2D.kernel.dtype: %s' % layer.kernel.dtype.name)
        output = layer.output
        print('{:<20s}.dtype: {:<10s}'.format(layer.name, output.dtype.name))


if __name__ == '__main__':
    model = model((32, 32, 1),  num_classes=200, max_value=False, input_thr=True)
    model.summary()
    # quant_aware_model = quantization_aware_training(model)
    # quant_aware_model.summary()

    # path = 'Keras_NN_profiter/models/your_model.h5'
    model_name = 'sample_model'
    print_file_model = open('statistics_files/{0}_report.txt'.format(model_name), 'w+')
    print('\n{0}\n'.format(model_name), file=print_file_model)
    Total_neurons, Total_FlOPS = NN_profiler(model, print_file_model)
    print('\nTotal number of Neurons: {:<10f} K'.format(np.round(Total_neurons / 10 ** 3, 2)), file=print_file_model)
    print('\nTotal number of operations: {:<10f} MFLOPs'.format(np.round(Total_FlOPS / 10 ** 6, 2)),
          file=print_file_model)
    total_param = output_nb_param(model)
    print('\nTotal number of Parameters: {:<10f} K'.format(np.round(total_param / 10 ** 3, 2)), file=print_file_model)
    print_file_model.close()


