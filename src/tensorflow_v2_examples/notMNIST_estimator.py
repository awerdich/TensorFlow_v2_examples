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

# Number of output classes
n_classes = 10

#%% The input function specifies how data is converted to a tf.data.Dataset that feeds the input pipeline

def make_input_fn(tfr_file_list, n_epochs = 1, shuffle = True, batch_size = 32):
    def input_function():

        ds = DatasetProvider(tfr_file_list,
                             output_height = im_size[0],
                             output_width = im_size[1],
                             n_epochs = n_epochs)
        ds = ds.make_batch(batch_size = batch_size, shuffle = shuffle)

        return ds
    return input_function

train_input_fn = make_input_fn([train_file], n_epochs = 1)
eval_input_fn = make_input_fn([valid_file], n_epochs = 1, shuffle = False)

#%% Inspect the data set
b_size = 10
ds = make_input_fn([train_file], batch_size = b_size)()

for im_batch, lab_batch in ds.take(1):

    fig, ax = plt.subplots(ncols = b_size, figsize = (b_size, 1.5))
    idx = np.random.randint(0, im_batch.shape[0], size = b_size)

    for i, ax1 in enumerate(ax):
        image = np.squeeze(im_batch[idx[i]])
        label = letter_dict[np.argmax(lab_batch[idx[i]])]

        print()
        print('Mean:', np.mean(image))
        print('Std:', np.std(image))
        print('Min:', np.min(image))
        print('Max:', np.max(image))

        ax1.imshow(image, cmap='gray')
        ax1.grid(False)
        ax1.axis('off')
        ax1.set_title(label)

    plt.show()

#%% Set up the model and estimator

# Build model
model = NotMNIST_model(im_size = im_size, n_outputs = n_classes).CNN()
# Show the summary
model.summary()

# Compile the model
model.compile(loss = tf.keras.losses.categorical_crossentropy,
              optimizer = tf.keras.optimizers.Adadelta(),
              metrics = ['accuracy'])


#%% Train the model

# Estimator configuration
est_config = tf.estimator.RunConfig(save_summary_steps = 100,
                                    save_checkpoints_steps = 100,
                                    keep_checkpoint_max = 10,
                                    log_step_count_steps = 100)

# Create the estimator from the keras model
model_fn = tf.keras.estimator.model_to_estimator(keras_model = model,
                                                 model_dir = log_dir,
                                                 config = est_config)

#%% Training and evaluation loop

train_steps_per_repeat = 200
n_repeats = 10

for r in range(n_repeats):

    print('Training run:', r+1)
    # The train_and_evaluate function seems to be broken in v2.0

    # Try a logging hook


    model_fn.train(train_input_fn, steps = train_steps_per_repeat)

    print('Running evaluation')
    model_fn.evaluate(eval_input_fn, steps = None)


#%% Evaluation




