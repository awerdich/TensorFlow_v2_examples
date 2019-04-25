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

# Define the estimator's input_fn
def input_fn(is_training, tfr_file_list, batch_size, n_epochs = 1, num_parallel_calls = 1):

    dataset = DatasetProvider(tfr_file_list,
                              output_height = im_size[0],
                              output_width = im_size[1],
                              n_epochs = 1)

    dataset = dataset.make_batch(batch_size = batch_size, shuffle = is_training)





    # This only works in eager mode
    #image_batch, label_batch = next(iter(dataset))

    # TF 1.X
    #iterator = dataset.make_one_shot_iterator()
    #image_batch, label_batch = iterator.get_next()

    # Set the shape of the output images
    #image_batch = tf.reshape(image_batch, shape=(-1, *im_size, 1))
    #one_hot_label_batch = tf.one_hot(label_batch, depth=10)

    #return image_batch, one_hot_label_batch

    return dataset

#%% Set up the model and estimator

# Build model
model = NotMNIST_model(im_size = (28, 28), n_outputs = 10).CNN()
# Show the summary
model.summary()

# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.optimizers.Adam(learning_rate = 1e-4),
              metrics = ['accuracy'])

# Estimator configuration
est_config = tf.estimator.RunConfig(save_summary_steps = 100,
                                    save_checkpoints_steps = 100,
                                    keep_checkpoint_max = 10,
                                    log_step_count_steps = 50)

# Create the estimator from keras model
notMNIST_estimator = tf.keras.estimator.model_to_estimator(keras_model = model,
                                                           model_dir = log_dir,
                                                           config = est_config)

# Specifications for training and evaluation
train_spec = tf.estimator.TrainSpec(input_fn = lambda: input_fn(is_training = True,
                                                        tfr_file_list = [train_file],
                                                        batch_size = 64,
                                                        n_epochs = 1,
                                                        num_parallel_calls = 1),
                                    max_steps = 600)

eval_spec = tf.estimator.EvalSpec(input_fn = lambda: input_fn(is_training = False,
                                                      tfr_file_list = [valid_file],
                                                      batch_size = 64,
                                                      n_epochs = 1,
                                                      num_parallel_calls = 1),
                                  steps = 100)

#%% Test the data pipeline
tfr_file = train_file

im_batch, lab_batch = input_fn(is_training = True,
                   tfr_file_list = [tfr_file],
                   batch_size = 8)

n = 6
fig, ax = plt.subplots(ncols = n, figsize = (2*n, 2))
idx = np.random.randint(0, im_batch.shape[0], size = n)


for i, ax1 in enumerate(ax):

    image = np.squeeze(im_batch[idx[i]])
    label = letter_dict[np.argmax(lab_batch[idx[i]])]

    ax1.imshow(image, cmap = 'gray')
    ax1.grid(False)
    ax1.axis('off')
    ax1.set_title(label)

plt.show()

#%% Run training and validation
tf.estimator.train_and_evaluate(notMNIST_estimator, train_spec, eval_spec)

#%% Use the keras training loop






