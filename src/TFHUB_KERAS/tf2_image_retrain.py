# Imports

import os
import numpy as np

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt


import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers, Model

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

#%% Parameters
im_size = (224, 224)
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

# All of TensorFlow Hub's image modules expect float inputs in the [0, 1] range
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=im_size)

#The resulting object is an iterator that returns image_batch, label_batch pairs.
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

#%% show an image
fig, ax = plt.subplots(figsize = (5, 5))
ax.imshow(image_batch[2])
plt.show()

#%% Set up the model

# HUB URL
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

# Create the feature extractor
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))

#%% Apply the feature extractor to an image batch
feature_batch = feature_extractor_layer(image_batch)

# Freeze the variables in the feature extractor layer, so that the training only modifies the new classifier layer
feature_extractor_layer.trainable = False

# Build a model with the feature extractor
def CNN(im_size = (224, 224)):

    input_layer = layers.Input(shape = (*im_size, 3), name = 'input')

    net = feature_extractor_layer(input_layer)
    net = layers.Dense(units = 5, activation = 'softmax')(net)

    return Model(inputs = input_layer, outputs = net)

model = CNN()
model.summary()

#%% Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

#%% Use custom callback to monitor loss
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats_callback])

#%% Run the image batch through the model
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)

#%% Export the model

import time
t = time.time()

export_path = "/tf/data/saved_models/{}".format(int(t))
tf.keras.experimental.export_saved_model(model, export_path)

export_path

#%% Reload model and compare the predictions with the old one.

reloaded = tf.keras.experimental.load_from_saved_model(export_path,
                                                       custom_objects={'KerasLayer':hub.KerasLayer})

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()
