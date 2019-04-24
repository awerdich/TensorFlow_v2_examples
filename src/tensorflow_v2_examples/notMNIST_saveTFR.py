#%% Imports
import os
import pickle
import numpy as np

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow_v2_examples.notMNIST_TFR_provider import Dset

print('Tensorflow version:', tf.__version__)

#%% Filenames and paths
data_root = os.path.normpath('/tf/data/notMNIST')
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

#%% Convert data into TFR

with open(pickle_file, mode = 'rb') as f:
    data = pickle.load(f)

dset_images = [s for s in data.keys() if s.split('_')[-1]=='dataset']
dset_labels = [d.split('_')[0]+'_labels' for d in dset_images]

for i, image_set in enumerate(dset_images):
    image_data = data[image_set]
    label_data = data[dset_labels[i]]
    TFRfilename = image_set.split('_')[0] + '.tfrecords'

    if os.path.exists(os.path.join(data_root, TFRfilename)):
        print('%s already present - Skipping.' % TFRfilename)
    else:
        dset_object = Dset(data_root)
        dset_object.create_tfr(TFRfilename, image_data, label_data)

#%% Check the size of the tfrecords data files
file_list = [file for file in os.listdir(data_root) if os.path.splitext(file)[-1]=='.tfrecords']

for file in file_list:
    statinfo = os.stat(os.path.join(data_root, file))
    print('TFRecords file {} size is {} MB'.format(file, np.round(statinfo.st_size/1e6)))
