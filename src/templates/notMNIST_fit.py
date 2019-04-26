# Implementation of the "TensorFlow 2.0 for experts" tutorial
# https://www.tensorflow.org/alpha/tutorials/quickstart/advanced

# Imports
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib
import string

matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

# Custom imports
from tensorflow_v2_examples.notMNIST_TFR_provider import DatasetProvider
from tensorflow_v2_examples.notMNIST_models import NotMNIST_model

print('TensorFlow version:', tf.__version__)

#%% Set up the model

# Build model
model = NotMNIST_model(im_size = (28, 28), n_outputs = 10).CNN()
# Show the summary
model.summary()

# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.optimizers.Adam(learning_rate = 1e-4),
              metrics = ['accuracy'])


#%% Input pipeline

data_root = os.path.normpath('/tf/data/notMNIST')
log_dir = os.path.join(data_root, 'log')
train_file = glob.glob(data_root +'/train*.tfrecords')[0]
test_file = glob.glob(data_root +'/test*.tfrecords')[0]
valid_file = glob.glob(data_root +'/valid*.tfrecords')[0]

# Translate numbers into letters
letter_dict = {n: ch for n, ch in enumerate(string.ascii_uppercase) if n<10}

# TFR image output size
im_size = (28, 28)

# Make a dataset
train_set = DatasetProvider([train_file],
                          output_height = im_size[0],
                          output_width = im_size[1],
                          n_epochs = 1)

train_set = train_set.make_batch(batch_size = 128, shuffle = True)

val_set = DatasetProvider([valid_file],
                          output_height = im_size[0],
                          output_width = im_size[1],
                          n_epochs = 1)

val_set = val_set.make_batch(batch_size = 128, shuffle = False)

#%% Look at the data
im_batch, lab_batch = next(iter(val_set))

n = 6
fig, ax = plt.subplots(ncols = n, figsize = (2*n, 2))
idx = np.random.randint(0, im_batch.shape[0], size = n)


for i, ax1 in enumerate(ax):

    image = np.squeeze(im_batch[idx[i]])
    label = letter_dict[np.argmax(lab_batch[idx[i]])]

    print('Mean:', np.mean(image),
          'Std:', np.std(image),
          'Min:', np.min(image),
          'Max:', np.max(image))


    ax1.imshow(image, cmap = 'gray')
    ax1.grid(False)
    ax1.axis('off')
    ax1.set_title(label)

plt.show()

#%% Training process
history = model.fit(train_set,
                    epochs = 1,
                    validation_data = val_set,
                    validation_steps = None)



