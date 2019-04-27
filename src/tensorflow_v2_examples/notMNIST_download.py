import sys
import os
import tarfile
import numpy as np
import pickle
import string
from six.moves.urllib.request import urlretrieve

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

#%% Path names and parameters
num_classes = 10
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
url = 'https://commondatastorage.googleapis.com/books1000/'
data_root = '/tf/data/notMNIST'
last_percent_reported = None
# Translate numbers into letters
letter_dict = {n: ch for n, ch in enumerate(string.ascii_uppercase) if n<10}

#%% Download the data archives

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def download_archive(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename

# Download the data archives from Google

# This training dataset contains 529,139 images
train_filename = download_archive('notMNIST_large.tar.gz', 247336696)

# This test dataset contains 18,746 images
test_filename = download_archive('notMNIST_small.tar.gz', 8458043)

#%% Extract the images into folders

def extract_to_folders(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

# Extract the images
train_folders = extract_to_folders(train_filename)
test_folders = extract_to_folders(test_filename)

#%% Save data sets as pickle files

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:

            # image_data = (ndimage.imread(image_file).astype(float) -
            # pixel_depth / 2) / pixel_depth

            # imread is deprecated in Scipy. Use plt.imread instead.
            # We also want to use unsigned int8 images to save space

            im = plt.imread(image_file)
            image_data = (im - np.min(im)) / (np.max(im) - np.min(im)) * pixel_depth
            image_data = image_data.astype(np.uint8)

            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))

            dataset[num_images, :, :] = image_data
            num_images = num_images + 1

        except:
            print('Could not read:', image_file, ':', '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]

    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))

    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

# Create data sets as pickle files
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

#%% Merge and prune the data

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.uint8)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

# Split large dataset into training and validation and then merge into data sets
# Available number of images is ~ 529,139
train_size = 400000
valid_size = 18000
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)

# Merge small data set into the test set
test_size = 15000
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)


#%% Remove empty images

def remove_empty(image_data, label_data):
    ''' Input: image_data [sample, rows, columns]
        label_data [label]
        Output: image_data_clean, label_data_clean, empty_idx'''

    empty_idx = [idx for idx in range(len(image_data)) if
                 (np.max(image_data[idx]) - np.min(image_data[idx])) < 0.5]

    not_empty_idx = [idx for idx in range(len(image_data)) if
                     idx not in (empty_idx)]

    if not not_empty_idx:

        print('All images are empty.')
        image_data_clean = image_data
        label_data_clean = label_data

    else:

        image_data_clean = np.stack([image_data[idx] for idx in not_empty_idx])
        label_data_clean = np.stack([label_data[idx] for idx in not_empty_idx])

    return image_data_clean, label_data_clean, empty_idx

# Remove empty images
train_dataset, train_labels, empty_train = remove_empty(train_dataset, train_labels)
print('Removed', len(empty_train), 'empty images from training set.')
test_dataset, test_labels, empty_test = remove_empty(test_dataset, test_labels)
print('Removed', len(empty_test), 'empty images from test set.')
valid_dataset, valid_labels, empty_valid = remove_empty(valid_dataset, valid_labels)
print('Removed', len(empty_valid), 'empty images from validation set.')

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

#%% Shuffle the data

def shuffle_data(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

# Randomize the data
train_dataset, train_labels = shuffle_data(train_dataset, train_labels)
test_dataset, test_labels = shuffle_data(test_dataset, test_labels)
valid_dataset, valid_labels = shuffle_data(valid_dataset, valid_labels)

#%% Save the data in one pickle file

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:{}'.format(np.round(statinfo.st_size/1e6)), 'MB')

#%% plot a few random images from the training set with label
#n = 6
#fig, ax = plt.subplots(nrows = 1, ncols = n, figsize = (2*n, 2))
#train_idx = np.random.randint(0, train_size, size = n)


#for i, ax1 in enumerate(ax):
#    ax1.imshow(train_dataset[train_idx[i]], cmap = 'gray')
#    ax1.grid(False)
#    ax1.axis('off')
#    ax1.set_title(letter_dict[train_labels[train_idx[i]]])

#plt.show()
