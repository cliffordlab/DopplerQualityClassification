from tensorflow.keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from skimage.transform import resize
import os
import random
    

def create_directory_if_not_exists(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        try:
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")


def plot_history(hist, num):
    # function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history['loss'], '.-')
    plt.plot(hist.history['val_loss'],'.-')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(f'history{num}.png')


class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim,return_coefficients=False,**kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        self.return_coefficients = return_coefficients        
        super(HierarchicalAttentionNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self._trainable_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask


    def call(self, x, mask=None):
            # size of x :[batch_size, sel_len, attention_dim]
            # size of u :[batch_size, attention_dim]
            # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))
        if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]
