# Keras models for notMIST dataset

import tensorflow as tf

# Keras imports
from tensorflow.keras import Input, layers, Model

#%% Model architecture (returns Keras model object)

class NotMNIST_model:
    '''Convolutional neural network for notMNIST classification
    data_root: folder with .tfrecords files
    input_dim: (28, 28, 1)
    output_dim: (10,)'''

    def __init__(self, im_size, n_outputs):
        self.im_size = im_size
        self.n_outputs = n_outputs

    def CNN(self, dropout=0.5):
        '''CNN_classifier for original notMNIST dataset.'''

        # Input layer
        input_net = Input(shape = (*self.im_size, 1), name = 'image')

        # Network
        net = layers.Conv2D(32, (3, 3), padding = 'same', activation = None)(input_net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(tf.nn.relu)(net)
        net = layers.MaxPool2D(pool_size=(2, 2), strides = 2)(net)

        net = layers.Conv2D(64, (3, 3), padding='same', activation=None)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(tf.nn.relu)(net)
        net = layers.MaxPool2D(pool_size = (2, 2), strides = 2)(net)

        # Dense layer
        net = layers.Conv2D(1024, (7, 7), padding = 'valid', activation = tf.nn.relu)(net)
        net = layers.Dropout(rate=dropout)(net)
        net = layers.Flatten()(net)

        # Logits layer
        net = layers.Dense(self.n_outputs, activation = 'softmax')(net)

        # Package layers in Keras model
        return Model(inputs = input_net, outputs = net)
