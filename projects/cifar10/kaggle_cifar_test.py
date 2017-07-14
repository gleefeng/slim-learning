# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets import cifar10
import tensorflow as tf
import numpy as np
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import dataset_utils
import csv
slim = tf.contrib.slim
BATCH_SIZE =100



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    checkpoint_path = tf.train.latest_checkpoint("./log/train")
    print(checkpoint_path)
    kaggle_test = "/tmp/cifar10"
    dataset = cifar10.get_split('kaggle', kaggle_test)
        
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset,shuffle=False)
    image, label = data_provider.get(['image', 'label'])
#    for item in dataset.list_items():
    
    preprocessing_fn = preprocessing_factory.get_preprocessing("cifarnet",is_training=False)
    image = preprocessing_fn(
        image,
        32,
        32)
    
    images, labels = tf.train.batch(
        [image, label],
        batch_size=BATCH_SIZE,
        allow_smaller_final_batch=True)
    
        # get the model prediction
    network_fn =nets_factory.get_network_fn("cifarnet",num_classes= 10,is_training=False)
    # run the image through the model
#    predictions,_ = lenet.lenet(images)
    predictions,_ = network_fn(images)
    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(predictions, 1))
   
    if dataset_utils.has_labels(kaggle_test):
        print("fffffffff")
        labels_to_names = dataset_utils.read_label_file(kaggle_test)
        print(labels_to_names)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess,checkpoint_path)
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        predicted_lables = np.zeros(300000)
        p=[]
        for i in range(1):
             print(i)
             predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] =predictions.eval()
        coord.request_stop()
        coord.join(threads)
        for i in range(300000):
            p.append(labels_to_names[predicted_lables[i]])
        np.savetxt('submission.csv', 
           np.c_[range(1,300000+1),predicted_lables], 
           delimiter=',', 
           header = 'id,Label', 
           comments = '', 
           fmt='%d')
        np.savetxt('s2.csv', 
           np.c_[range(1,300000+1),p], 
           delimiter=',', 
           header = 'id,Label', 
           comments = '', 
           fmt='%s')
#        with open("s3.csv","w") as f:
#            w =csv.writer(f)
#            w.writerows(zip(range(1,300000+1),p))
