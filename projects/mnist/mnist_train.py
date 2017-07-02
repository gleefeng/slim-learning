import tensorflow as tf
from datasets import mnist
from nets import lenet
from model import load_batch
#
slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_batches', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS

def main(args):
    # load the dataset
    dataset = mnist.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)

    # run the image through the model
    predictions,_ = lenet.lenet(images)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    tf.losses.softmax_cross_entropy(one_hot_labels,predictions)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    # use RMSProp to optimize
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer)

    # run training
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        save_summaries_secs=20,
        save_interval_secs =60)


if __name__ == '__main__':
    tf.app.run()
