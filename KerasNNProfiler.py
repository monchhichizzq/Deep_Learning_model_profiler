# -*- coding: utf-8 -*-
# @Time    : 2020/5/2 18:42
# @Author  : Zeqi@@
# @FileName: KerasNNProfiler.py
# @Software: PyCharm

import numpy as np
import keras.backend as K
from keras.models import load_model

def profile_conv2d(layer, print_save):
    Bias_add = 1 if layer.use_bias else 0
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[1:])
    k = layer.kernel_size
    # (Cin*(2*k^2+1) + (Cin-1)) * M * M * Cout
    # # 2*H*W*(Cin*K^2+1)*Cout
    conv_flops = input_shape[3]*k[0]*k[1] + Bias_add
    FLOPs = 2*conv_flops*np.prod(output_shape[1:])
    # step_1 = (input_shape[3] * (k[0] * k[1] + Bias_add) + (input_shape[3] - 1))
    # step_2 = np.prod(output_shape[1:])/10 ** 6
    # FLOPs = step_1*step_2*10**6
    # FLOPs = (input_shape[3]*k[0]*k[1])*np.prod(output_shape[1:])
    print('=====>>>[ {0}: {1} Neurons  {2} MFLOPs input:{3} output:{4} kernel:{5}] '.
          format(layer.__class__.__name__, nb_neurons_per_layer, FLOPs/10**6, input_shape[1:], output_shape[1:], k),file=print_save)
    return nb_neurons_per_layer, FLOPs


def profile_dense(layer, print_save):
    Bias_add = 1 if layer.use_bias else 0
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(input_shape[1:])
    # (2*I-1)*O # I is the input dimensionality and O is the output dimensitonality
    FLOPs = (input_shape[1] - 1 + Bias_add) * output_shape[1]
    print('=====>>>[ {0}: {1} Neurons  {2} MFLOPs ]'.format(layer.name, nb_neurons_per_layer, FLOPs / 10 ** 6),
          file=print_save)
    return nb_neurons_per_layer, FLOPs


def profile_linear(layer, print_save):
    input_shape = layer.input_shape
    nb_neurons_per_layer = np.prod(input_shape[1:])
    # (multiplication+addition)*output_shape
    FLOPs = 2 * np.prod(input_shape[1:])
    print('=====>>>[ {0}: {1} Neurons  {2} MFLOPs ]'.format(layer.name, nb_neurons_per_layer, FLOPs / 10 ** 6),
          file=print_save)
    return nb_neurons_per_layer, FLOPs


def profile_BatchNorm2d(layer, print_save):
    # BN can be regarded as linear
    # Batch size is variate, ops is not fixed
    # For one input smaple, gamma*in + beta == 2*input_shape ops
    input_shape = layer.input_shape
    FLOPs = 2 * np.prod(input_shape[1:])
    nb_neurons_per_layer = np.prod(input_shape[1:])


def profile_maxpool(layer, print_save):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    k = np.sqrt(np.prod(input_shape[1:])/ np.prod(output_shape[1:]))
    FLOPs = (k*k-1)*np.prod(layer.output_shape[1:])
    nb_neurons_per_layer = np.prod(output_shape[1:])
    print('=====>>>[ {0}: {1} Neurons  {2} MFLOPs input:{3} output:{4} kernel:{5}]'
          .format(layer.name, nb_neurons_per_layer, FLOPs / 10 ** 6, input_shape[1:], output_shape[1:],
                  (int(k), int(k))),
          file=print_save)
    return nb_neurons_per_layer, FLOPs


def profile_avgpool(layer, print_save):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    k = np.sqrt(np.prod(input_shape[1:]) / np.prod(output_shape[1:]))
    FLOPs = ((k * k - 1) + 1)* np.prod(layer.output_shape[1:])
    nb_neurons_per_layer = np.prod(output_shape[1:])
    print('=====>>>[ {0}: {1} Neurons  {2} MFLOPs input:{3} output:{4} kernel:{5}]'
          .format(layer.name, nb_neurons_per_layer, FLOPs / 10 ** 6, input_shape[1:], output_shape[1:], (int(k), int(k))),
          file=print_save)
    return nb_neurons_per_layer, FLOPs

def profile_Input(layer, print_save):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[1:])
    print('=====>>>[ {0}: {1} Neurons  ]'.format(layer.name, nb_neurons_per_layer),
          file=print_save)
    return nb_neurons_per_layer




def NN_profiler(model):
    Total_neurons, Total_FlOPS = 0, 0
    for layer in model.layers:
        # print('=====>>>[ {0}] '.format(layer.__class__.__name__), file=print_file_model)
        if layer.__class__.__name__ == 'InputLayer':
            nb_neurons_per_layer = profile_Input(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
        elif layer.__class__.__name__ == 'Conv2D':
            nb_neurons_per_layer, FLOPs =  profile_conv2d(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
            Total_FlOPS += FLOPs
        elif layer.__class__.__name__ == 'Dense':
            nb_neurons_per_layer, FLOPs = profile_dense(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
            Total_FlOPS += FLOPs
        elif layer.__class__.__name__ == 'MaxPooling2D':
            nb_neurons_per_layer, FLOPs = profile_maxpool(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
            Total_FlOPS += FLOPs
        elif layer.__class__.__name__ == 'AveragePooling2D':
            nb_neurons_per_layer, FLOPs = profile_avgpool(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
            Total_FlOPS += FLOPs
        elif layer.__class__.__name__ == 'GlobalAveragePooling2D':
            nb_neurons_per_layer, FLOPs = profile_avgpool(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
            Total_FlOPS += FLOPs


    return Total_neurons, Total_FlOPS


def output_nb_param(model):
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])



def test(model_name, path):
    # from keras.applications.vgg16 import VGG16
    # model = VGG16(weights='imagenet', include_top=True)
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(weights='imagenet', include_top=True)

    # model = load_model(path)
    model.summary()
    print('\n{0}\n'.format(model_name), file=print_file_model)
    Total_neurons, Total_FlOPS = NN_profiler(model)
    print('\nTotal number of Neurons: {0} M'.format(np.round(Total_neurons/10**6, 2)),  file=print_file_model)
    print('\nTotal number of operations: {0} GFLOPs'.format(np.round(Total_FlOPS / 10 ** 9, 2)), file=print_file_model)
    total_param = output_nb_param(model)
    print('\nTotal number of Parameters: {0} M'.format(np.round(total_param/10**6,2)), file=print_file_model)

if __name__ == '__main__':
    model_name = 'ResNet50'
    path = 'models/your_model.h5'
    print_file_model = open('statistics_files/{0}_report.txt'.format(model_name), 'w+')
    test(model_name, path)


# Reference
# Ref: Molchanov P, Tyree S, Karras T, et al. Pruning Convolutional Neural Networks for Resource Efficient Inference[J]. 2016