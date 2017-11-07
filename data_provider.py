'''
Utilities for data provision
'''

from os.path import join as opj

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

def prepare_sample_list(data_root, sample_list_file, classes=None):
    '''
    Prepare list of image paths and corresponding labels
    -> data_root: root folder for images (prefix)
    -> sample_list_file: path to a file with image list
    -> classes: list of class labels to prepare a subset
    <- (list of image paths, list of labels)
    '''
    with open(sample_list_file) as f:
        sample_lines = f.readlines()

    image_list = []
    label_list = []
    for line in sample_lines:
        filename, label = line.split(' ')[:2]
        label = int(label)
        if classes is None or label in classes:
            image_list.append(opj(data_root, filename))
            if classes is not None:
                label = classes.index(label)
            label_list.append(label)

    return image_list, label_list

def queue_data_batch(image_list, label_list, batch_size=1, crop_size=(256, 256)):
    '''
    Construct batched TF queues for input images and GT labels
    -> image_list: list of image paths
    -> label_list: list of labels
    -> batch_size: batch size
    -> crop_size: all images are padded with zeros and randomly cropped
    <- (batched image queue, batched label queue)
    '''
    with tf.name_scope('input_data'):
        images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
        labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

        input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

        channels = 3

        label = input_queue[1]
        image_file = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_file, channels=channels)
        image = tf.cast(image, tf.float32)
        image = tf.image.pad_to_bounding_box(image, 0, 0,
                                            tf.maximum(crop_size[0], tf.shape(image)[0]),
                                            tf.maximum(crop_size[1], tf.shape(image)[1]),)
        image = tf.random_crop(image, (crop_size[0], crop_size[1], channels))

        image_batch, label_batch = tf.train.batch([image, label],
                                                    batch_size=batch_size)

    return image_batch, label_batch

def imagenet_data(data_root, sample_list_file, batch_size=1, crop_size=(256, 256), classes=None):
    '''
    Prepare batched queues of ImageNet data (images and labels)
    -> data_root: root folder for images (prefix)
    -> sample_list_file: path to a file with image list
    -> batch_size: batch size
    -> crop_size: all images are padded with zeros and randomly cropped
    -> classes: list of class labels to prepare a subset
    <- (batched image queue, batched label queue, number of samples)
    '''
    image_list, label_list = prepare_sample_list(
        data_root,
        sample_list_file,
        classes=classes)

    num_samples = len(image_list)

    image, label = queue_data_batch(
        image_list,
        label_list,
        batch_size=batch_size,
        crop_size=crop_size)

    return image, label, num_samples


