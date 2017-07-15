# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:41:42 2017

@author: Administrator
"""

import tensorflow as tf
import pandas as pd
import numpy as np
slim = tf.contrib.slim
from nets import nets_factory
from preprocessing import lenet_preprocessing
from nets import lenet
import matplotlib.pyplot as plt

# Convert class labels from scalars to one-hot vectors 
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def read_traincsv():
#    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

#    image, label = data_provider.get(['image', 'label'])
     data = pd.read_csv("./data/train.csv")
     images = data.iloc[:,1:].values
     images = images.astype(np.float32)
     images = images.reshape(images.shape[0],28,28,1)
     labels_flat = data.iloc[:,0].values.ravel()
     labels_count = np.unique(labels_flat).shape[0]
     labels = dense_to_one_hot(labels_flat, labels_count)
     labels = labels.astype(np.int64)
     images = images-128.0
#     print(images[0])
     images = images/128.0 
#     print(images[0])
#     plt.imshow(images[0])
     
     return images, labels 
def read_testcsv():
#    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

#    image, label = data_provider.get(['image', 'label'])
    
     test_images = pd.read_csv('./data/test.csv').values

     images = test_images.astype(np.float32)
     images = images.reshape(images.shape[0],28,28,1)
     images = images-128.0
#     print(images[0])
     images = images/128.0 
#     print(images[0])
#     plt.imshow(images[0])

     return images 
    
def show_train_accuracies():
    BATCH_SIZE = 512
    train_accuracies=[]
    tf.logging.set_verbosity(tf.logging.DEBUG)
    checkpoint_path = tf.train.latest_checkpoint("./log/train")
    print(checkpoint_path)
    images, labels = read_traincsv()
    print(labels.shape)
    images_x = tf.placeholder("float",[None,28,28,1])
    labels_y = tf.placeholder("int64",[None,10])
    predictions,_ = lenet.lenet(images_x)
    predictions = tf.to_int64(tf.argmax(predictions, 1))
    
    
    correct_predict = tf.equal(predictions, tf.argmax(labels_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
         tf.global_variables_initializer().run()
         saver.restore(sess,checkpoint_path)
         for i in range(images.shape[0]//BATCH_SIZE):
#             p =predictions.eval(feed_dict={images_x: images[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
#             print(p)
              train_accuracy = accuracy.eval(feed_dict={images_x: images[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                  labels_y: labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
              train_accuracies.append(train_accuracy)
         av_train_accuracies = sum(train_accuracies)/len(train_accuracies)
         print("av_train_accuracies %f" % av_train_accuracies)
    sess.close()

def save_test():
    BATCH_SIZE = 1
    tf.logging.set_verbosity(tf.logging.DEBUG)
    checkpoint_path = tf.train.latest_checkpoint("./log/train")
    print(checkpoint_path)
    images = read_testcsv()
    print(images.shape)
    images_x = tf.placeholder("float",[None,28,28,1])
    predictions,_ = lenet.lenet(images_x)
    predictions = tf.to_int64(tf.argmax(predictions, 1))
    predicted_lables = np.zeros(images.shape[0])    
    saver = tf.train.Saver()
    with tf.Session() as sess:
         tf.global_variables_initializer().run()
         saver.restore(sess,checkpoint_path)
         for i in range(images.shape[0]):
             predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] =predictions.eval(feed_dict={images_x: images[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
         np.savetxt('submission.csv', 
           np.c_[range(1,len(images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
    sess.close()



if __name__=='__main__':
#    tf.app.run()
     save_test()
         
