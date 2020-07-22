# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 11:59
# @Author  : Zeqi@@
# @FileName: Custom_block.py.py
# @Software: PyCharm

import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Layer,BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling1D, AveragePooling1D
from tensorflow.keras import regularizers




# Inference
class Customblock(Model):
    def __init__(self,
                 filters=16,
                 kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-3),
                 **kwargs):
        super(Customblock, self).__init__(**kwargs)

        self.conv = Conv2D(filters, 
                            kernel_size, 
                            strides=1, 
                            padding='same', 
                            activation=None, 
                            use_bias=False,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)

        if activation is not None:
            self.act = Activation(activation)
        self.bn = BatchNormalization()

    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.act(x)
        outputs = self.bn(x, training=training)
        return outputs

    def get_config(self):
        config = {"filters": self.filters}
        base_config = super(SparnetConv2D, self).get_config()
        print(base_config)
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    # Example
    input = Input(shape=(32, 32, 1))
    x = Customblock(16, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-3))(input, training=True)
    x = Customblock(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-3))(x, training=False)
    output = Customblock(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-3))(x, training=True)
    model = Model(inputs=input, outputs=output)

    model.summary()

    for layer in model.layers:
        layer.trainable = False
        print(layer.trainable)

    W = model.get_weights()
    print(len(W))
    for i in range(len(W)):
        print(np.shape(W[i]))


# 数据测试
# import tensorflow as tf
# print(layer(tf.ones([1,3,9,9])))
# 查看网络中的变量名
# print([x.name for x in resnetBlock.trainable_variables])