# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets import cifar10
import tensorflow as tf
import os
import sys
from datasets import dataset_utils
import matplotlib.pyplot as plt
import pandas as pd
slim = tf.contrib.slim
SPLITS_TO_SIZES = {'train': 2295, 'test': 1531}
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A  color image.',
    'label': 'A single integer between 0 and 1',
}
_FILE_PATTERN = 'InS_%s.tfrecord'
def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if not reader:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/height': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/width':tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),    
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'height': slim.tfexample_decoder.Tensor('image/height'),
      'width': slim.tfexample_decoder.Tensor('image/width'),
                                            
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      num_classes=2,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)
def show_img():
    kaggle_test = "/kaggleData/InvasiveSpeciesMonitoring"
    dataset = get_split('train', kaggle_test)
    print(dataset)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset,shuffle=False)
    image, label,height,width = data_provider.get(['image', 'label','height','width'])
    image.set_shape([866,1154,3])
    images, labels = tf.train.batch(
        [image, label],
        batch_size=1,
        allow_smaller_final_batch=True)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        images_ ,labels_=sess.run([images,labels])
        print(images_.shape)
        images_ =tf.squeeze(images_).eval()
        print(images_.shape,images_.dtype)
        print(labels_)
        plt.imshow(images_)
        coord.request_stop()
        coord.join(threads)

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def convert_dataset(photo_filenames, class_names, output_filename):

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          for i in range(len(photo_filenames)):
            sys.stdout.write('\r>> Converting image %d/%d '%(i+1, len(photo_filenames)))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(photo_filenames[i], 'rb').read()
            class_name = class_names[i]
            height, width = image_reader.read_image_dims(sess, image_data)
            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_name)
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()
def createTrainData(datadir):
    output_filename = os.path.join(datadir,"InS_train.tfrecord")
    train_data = os.path.join(datadir,"train")
    train_label = os.path.join(datadir,"train_labels.csv")
    data = pd.read_csv(train_label)
    print("has %d jpg" %len(data))
    photo_filenames = []
    class_names = []
    for i in range(len(data)):
        value = data.loc[i]
        picname = os.path.join(train_data,(value["name"].astype(str)+".jpg"))
        if not os.path.isfile(picname):
            print("not exist the file!\n")
            assert(0)
        photo_filenames.append(picname)
        class_names.append(value["invasive"])
    convert_dataset(photo_filenames, class_names, output_filename)
#    print(photo_filenames)
def createTestData(datadir):
    test_data = os.path.join(datadir,"test")
    output_filename = os.path.join(datadir,"InS_test.tfrecord")
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(1531):
                    picname =os.path.join(test_data,(str(i+1)+".jpg"))
                    sys.stdout.write('\r>> Converting image %s '%(picname))
                    sys.stdout.flush()
                    if not os.path.isfile(picname):
                        print("not exist the file!\n")
                        assert(0)
                    # Read the filename:
                    image_data = tf.gfile.FastGFile(picname, 'rb').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, 0)
                    tfrecord_writer.write(example.SerializeToString())    
    sys.stdout.write('\n')
    sys.stdout.flush()
            
def createData(datadir):
    createTrainData(datadir)
    createTestData(datadir)
    
        
    
if __name__ == "__main__":
#    show_img()
#    createData("/kaggleData/InvasiveSpeciesMonitoring")
