# Model_profier
A tool helps you to calculate the Neuron numbers, Parameters, Op numbers of DL model 

## Library
- Keras 2.4.0
- Tensorflow 2.2.0

## Layers currently supported:
- Conv2d
- Linear
- Batch Normalization2D
- Max-pooling2D
- Average-pooling2D
- Dense 

## Test
 - You can test the scripts with your pretrained model and save its statistics
 - You can test on the standard CNN model  VGG16/ ResNet50/..
 - Build your own CNN model in `models.py`


## Output 
All the model information will be recorded in the '*.txt' in the 'statistics_files' folder. 

## FLOPs computation
To compute the number of floating-point operations (FLOPs), we assume convolution is implemented as a sliding window and that the nonlinearity function is computed for free.

Reference :
 Molchanov P, Tyree S, Karras T, et al. Pruning Convolutional Neural Networks for Resource Efficient Inference[J]. 2016

## To do list
- RNN Layer
- Conv3d
- Depthwise Convolution
- Depthwise Separable Convolution
