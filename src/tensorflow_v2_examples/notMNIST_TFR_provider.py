#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 23:24:38 2018

Some helper functions for processing noMNIST data sets

@author: andy
"""
#%% Imports
import os
import sys
import numpy as np
import tensorflow as tf

from pdb import set_trace

#%% Functions and classes

class Dset:
    
    def __init__(self, data_root):
        self.data_root = data_root
    
    def create_tfr(self, filename, image_data, label_data):
        ''' Build a TFRecoreds data set from numpy arrays'''
        
        file = os.path.join(self.data_root, filename)

        with tf.io.TFRecordWriter(file) as writer:
            
            print('Converting:', filename)
            n_images = len(image_data)
            
            for i in range(n_images):
                
                # Print the percentage-progress.
                self._print_progress(count = i, total = n_images-1)
        
                im_bytes = image_data[i].astype(np.uint16).tobytes()
                im_shape_bytes = np.array(image_data[i].shape).astype(np.uint16).tobytes()
                im_label = label_data[i]
    
                # Build example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': self._wrap_bytes(im_bytes),
                    'shape': self._wrap_bytes(im_shape_bytes),
                    'label': self._wrap_int64(im_label)}))
    
                # Serialize example
                serialized = example.SerializeToString()
    
                # Write example to disk
                writer.write(serialized)
            
    def list_tfr(self, data_path, tfrext = '.tfrecords'):
        ''' Return a list of TFRecords data files in path'''
        
        file_list = os.listdir(data_path)
        tfr_list = [os.path.join(data_path, f) for f in file_list if os.path.splitext(f)[-1] == tfrext]
        
        return tfr_list
    
    # Integer data (labels)
    def _wrap_int64(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # Byte data (images, arrays)
    def _wrap_bytes(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _wrap_float(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    # Progress update
    def _print_progress(self, count, total):
        pct_complete = float(count) / total

        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {0:.1%}".format(pct_complete)

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

class DatasetProvider:
    ''' Creates a dataset from a list of .tfrecords files.'''
    
    def __init__(self, tfr_file_list, output_height = 28, output_width = 28, n_epochs = 1):
        self.tfr_file_list = tfr_file_list
        self.output_height = output_height
        self.output_width = output_width
        self.n_epochs = n_epochs
        
    def _parse(self, serialized):
        
        example = {'image': tf.io.FixedLenFeature([], tf.string),
                   'shape': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)}
    
        # Extract example from the data record
        example = tf.io.parse_single_example(serialized, example)
    
        # Convert image to tensor and shape it
        image_raw = tf.io.decode_raw(example['image'], tf.uint16)
        shape = tf.io.decode_raw(example['shape'], tf.uint16)
        shape = tf.cast(shape, tf.int32) # tf.reshape requires int16 or int32 types
        image = tf.reshape(image_raw, shape)
        
        # Add color channel
        image = tf.expand_dims(image, 2)
        
        # Resize and crop image if needed
        image = tf.image.resize_image_with_crop_or_pad(image, 
                                                       self.output_height, 
                                                       self.output_width)
        
        # Convert to float
        image = tf.cast(image, tf.float64)
        
        # Linearly scale image to have zero mean and unit norm
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, shape = (self.output_height, self.output_width, 1))
        
        label = example['label']
        one_hot_label = tf.one_hot(label, depth=10)
    
        return image, one_hot_label
    
    def make_batch(self, batch_size, shuffle):
        
        dataset = tf.data.TFRecordDataset(self.tfr_file_list)
        
        # Shuffle data 
        if shuffle:
            dataset = dataset.shuffle(buffer_size = 2 * batch_size,
                                      reshuffle_each_iteration = True)
        
        # Parse records
        dataset = dataset.map(self._parse)
        
        # Batch it up
        dataset = dataset.batch(batch_size)
        
        return dataset.repeat(self.n_epochs)
