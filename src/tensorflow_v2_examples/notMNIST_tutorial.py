# Implementation of the "TensorFlow 2.0 for experts" tutorial
# https://www.tensorflow.org/alpha/tutorials/quickstart/advanced

# Imports
import os
import glob
import tensorflow as tf
import matplotlib

matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

# Custom imports
from tensorflow_v2_examples.notMNIST_TFR_provider import DatasetProvider
from tensorflow_v2_examples.notMNIST_models import NotMNIST_model

print('TensorFlow version:', tf.__version__)

#%% Load data set
data_root = os.path.normpath('/tf/data/notMNIST')
train_file = glob.glob(data_root +'/train*.tfrecords')[0]
test_file = glob.glob(data_root +'/test*.tfrecords')[0]
valid_file = glob.glob(data_root +'/valid*.tfrecords')[0]


# Create data set
mnist_train = DatasetProvider([train_file]).make_batch(batch_size = 32, shuffle = True)
mnist_test = DatasetProvider([test_file]).make_batch(batch_size = 32, shuffle = False)

# Build model
model = NotMNIST_model(im_size = (28, 28), n_outputs = 10).CNN()
# Show the summary
model.summary()

# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.optimizers.Adam(learning_rate = 1e-4),
              metrics = ['accuracy'])

# Estimator configuration
est_config = tf.estimator.RunConfig(save_summary_steps = 10,
                                    save_checkpoints_steps = 20,
                                    keep_checkpoint_max = 5,
                                    log_step_count_steps = 50)

#%% Plot some sample images
batch_image, batch_label = next(iter(mnist_train))



