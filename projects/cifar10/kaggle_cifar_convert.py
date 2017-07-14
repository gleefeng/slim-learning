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
slim = tf.contrib.slim

def show_img():
    kaggle_test = "/tmp/cifar10"
    dataset = cifar10.get_split('kaggle', kaggle_test)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset,shuffle=False)
    image, label = data_provider.get(['image', 'label'])
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
        print(images_,labels_)
        plt.imshow(images_)
        coord.request_stop()
        coord.join(threads)

def run(datadir,dstdir):
    kaggle_test = os.path.join(dstdir,"kaggle_test.tfrecord")
    if not tf.gfile.Exists(dstdir):
        tf.gfile.MakeDirs(dstdir)
    if tf.gfile.Exists(kaggle_test):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    image_filename = tf.placeholder(dtype=tf.string)
    image_file = tf.read_file(image_filename)
#        image_placeholder = tf.placeholder(dtype=tf.uint8)
#        encoded_image = tf.image.encode_png(image_placeholder)
    tfrecord_writer = tf.python_io.TFRecordWriter(kaggle_test)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("begin")
        for j in range(1,300001):
            filename = os.path.join(datadir,("%d.png" %j))
            sys.stdout.write('\r>> Reading %s' %filename)
            sys.stdout.flush()
#            statinfo = os.stat(filename)
#            print(statinfo)       
            png_string = sess.run(image_file, feed_dict={image_filename: filename})
            example = dataset_utils.image_to_tfexample(
            png_string, 'png'.encode(), 32, 32, 0)
            tfrecord_writer.write(example.SerializeToString())
        tfrecord_writer.close()
        print("over create %s\n",kaggle_test)
#                statinfo = os.stat(filepath)  
#            image = np.squeeze(images[j]).transpose((1, 2, 0))


#            png_string = sess.run(encoded_image,
#                                  feed_dict={image_placeholder: image})
            

#            example = dataset_utils.image_to_tfexample(
#            png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
#            tfrecord_writer.write(example.SerializeToString())
        
    
if __name__ == "__main__":
#    run("/tmp/test","/tmp/kaggle_test")
    show_img()
