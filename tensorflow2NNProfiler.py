# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 22:51
# @Author  : Zeqi@@
# @FileName: tensorflow2NNProfiler.py
# @Software: PyCharm


import os
import argparse
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Output the number of model parameters, the number of consumed neurons and the operation numbers of the target model')
    parser.add_argument('--model_name', default='ResNet50',
                        help="Set the name of the target model, e.g. ResNet50, VGG16")
    parser.add_argument('--model_path', default='models/example_model.h5',
                        help="Import the target model, e.g. example_model.h5")
    args = parser.parse_args()
    return args

def profile_conv2d(layer, print_save):
    Bias_add = 1 if layer.use_bias else 0
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[1:])
    k = layer.kernel_size
    # (Cin*(2*k^2+1) + (Cin-1)) * M * M * Cout
    # # 2*H*W*(Cin*K^2+1)*Cout
    conv_flops = np.float64(input_shape[3]*k[0]*k[1] + Bias_add)
    output_shape_prod= np.float64(np.prod(output_shape[1:]))
    FLOPs = 2*conv_flops*output_shape_prod
    # step_1 = (input_shape[3] * (k[0] * k[1] + Bias_add) + (input_shape[3] - 1))
    # step_2 = np.prod(output_shape[1:])/10 ** 6
    # FLOPs = step_1*step_2*10**6
    # FLOPs = (input_shape[3]*k[0]*k[1])*np.prod(output_shape[1:])
    print('=====>>>[ {:<20s}: {:<10d} Neurons  {:<15f} MFLOPs  input:{:<20s}  output:{:<20s}  kernel:{:<10s}]'.
          format(layer.__class__.__name__, nb_neurons_per_layer, np.round(FLOPs/10**6, 2), str(input_shape[1:]), str(output_shape[1:]), str(k)),file=print_save)
    return nb_neurons_per_layer, FLOPs

def profile_dense(layer, print_save):
    Bias_add = 1 if layer.use_bias else 0
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(input_shape[1:])
    # (2*I-1)*O # I is the input dimensionality and O is the output dimensitonality
    FLOPs = (input_shape[1] - 1 + Bias_add) * output_shape[1]
    print('=====>>>[ {:<20s}: {:<10d} Neurons  {:<15f} MFLOPs]'.format(layer.name, nb_neurons_per_layer,  np.round(FLOPs/10**6, 2)),
          file=print_save)
    return nb_neurons_per_layer, FLOPs

def profile_linear(layer, print_save):
    input_shape = layer.input_shape
    nb_neurons_per_layer = np.prod(input_shape[1:])
    # (multiplication+addition)*output_shape
    FLOPs = 2 * np.prod(input_shape[1:])
    print('=====>>>[ {:<20s}: {:<10d} Neurons  {:<15f} MFLOPs]'.format(layer.name, nb_neurons_per_layer,  np.round(FLOPs/10**6, 2)),
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
    print('=====>>>[ {:<20s}: {:<10d} Neurons  {:<15f} MFLOPs  input:{:<20s}  output:{:<20s}  kernel:{:<10s}]'
          .format(layer.name, nb_neurons_per_layer,  np.round(FLOPs/10**6, 2), str(input_shape[1:]), str(output_shape[1:]),
                  str((int(k), int(k)))),
          file=print_save)
    return nb_neurons_per_layer, FLOPs

def profile_avgpool(layer, print_save):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    k = np.sqrt(np.prod(input_shape[1:]) / np.prod(output_shape[1:]))
    FLOPs = ((k * k - 1) + 1)* np.prod(layer.output_shape[1:])
    nb_neurons_per_layer = np.prod(output_shape[1:])
    print('=====>>>[ {:<20s}: {:<10d} Neurons  {:<15f} MFLOPs  input:{:<20s}  output:{:<20s}  kernel:{:<10s}]'
          .format(layer.name, nb_neurons_per_layer, np.round(FLOPs/10**6, 2), str(input_shape[1:]), str(output_shape[1:]), str((int(k), int(k)))),
          file=print_save)
    return nb_neurons_per_layer, FLOPs

def profile_Input(layer, print_save):
    input_shape, output_shape = layer.input_shape, layer.output_shape
    nb_neurons_per_layer = np.prod(output_shape[0][1:])
    print(output_shape)
    print('=====>>>[ {:<20s}: {:<10d} Neurons ]'.format(layer.name, nb_neurons_per_layer),
          file=print_save)
    return nb_neurons_per_layer


def NN_profiler(model, print_file_model):
    Total_neurons, Total_FlOPS = 0, 0
    for layer in model.layers:
        # print('=====>>>[ {0}] '.format(layer.__class__.__name__), file=print_file_model)
        if layer.__class__.__name__ == 'InputLayer':
            nb_neurons_per_layer = profile_Input(layer, print_file_model)
            Total_neurons += nb_neurons_per_layer
        elif layer.__class__.__name__ == 'Conv2D':
            nb_neurons_per_layer, FLOPs = profile_conv2d(layer, print_file_model)
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
        else:
            print("Error: Layer == {} is not supported".format(layer.__class__.__name__))


    return Total_neurons, Total_FlOPS

def output_nb_param(model):
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

def test(model_name, path):
    if model_name == 'VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16
        model = VGG16(weights='imagenet', include_top=True)

    elif model_name == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import ResNet50
        model = ResNet50(weights='imagenet', include_top=True)

    else:
        model = load_model(path)

    model.summary()
    print('\n{0}\n'.format(model_name), file=print_file_model)
    Total_neurons, Total_FlOPS = NN_profiler(model, print_file_model)
    print('\nTotal number of Neurons: {0} M'.format(np.round(Total_neurons/10**6, 2)),  file=print_file_model)
    print('Total number of operations: {0} GFLOPs'.format(np.round(Total_FlOPS / 10 ** 9, 2)), file=print_file_model)
    total_param = output_nb_param(model)
    print('Total number of Parameters: {0} M'.format(np.round(total_param/10**6,2)), file=print_file_model)
    print_file_model.close()

if __name__ == '__main__':
    args = parse_arguments()

    model_name = args.model_name
    path = args.model_path

    os.makedirs('statistics_files', exist_ok=True)
    print_file_model = open('statistics_files/{0}_report.txt'.format(model_name), 'w+')
    test(model_name, path)


# Reference
# Ref: Molchanov P, Tyree S, Karras T, et al. Pruning Convolutional Neural Networks for Resource Efficient Inference[J]. 2016

# python tensorflow2NNProfiler.py --model_name=VGG16 --model_path=models/example_model.h5