import os
import cv2
import numpy as np
import tensorflow as tf

DATA_DIR='../../datasets/CASIA-WebFace'

### Functions to convert values to compatible types with tf.train.Example ###
def _bytes_feature(value):
    if(isinstance(value, type(tf.constant(0)))):
        value = value.numpy() 

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

### Writing the data to tfrecord ###
def write_to_tfrecord(data_dir, record_file='data/data.tfrecord', n_classes=1000):
    current_class=0
    with tf.io.TFRecordWriter(record_file) as writer:
        for (dir_, dirs, files) in os.walk(data_dir):
            if(current_class + 1 >= n_classes):
                break

            for file_ in files:
                abs_path = os.path.join(dir_, file_)
                print('[INFO] File %s processed, identity #%d ...' % (abs_path, current_class))

                feature = {
                    'image/filename' : _bytes_feature(abs_path.encode()),
                    'image/encoded'  : _bytes_feature(open(abs_path, 'rb').read()),
                    'image/source_id': _int64_feature(current_class)
                }

                ### Create example and write ###
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())

            current_class += 1

### Reading the data from tfrecord ###
def _parse_tfrecord(example_proto):
    feature_description = {
        'image/filename' : tf.io.FixedLenFeature([], tf.string),
        'image/encoded'  : tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.int64)
    }
    
    x = tf.io.parse_single_example(example_proto, feature_description)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (128,128))
    y_train = tf.one_hot(x['image/source_id'], 1000)

    return x_train, y_train

def read_from_tfrecord(record_file, batch_size=64, buffer_size=400000):
    dataset_len = int(sum(1 for _ in tf.data.TFRecordDataset(record_file)))

    raw_dataset = tf.data.TFRecordDataset(record_file)
    raw_dataset = raw_dataset.repeat()
    raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)

    dataset = raw_dataset.map(_parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset_len, dataset

