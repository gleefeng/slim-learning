import tensorflow as tf
from datasets import cifar10
from preprocessing import preprocessing_factory
from nets import nets_factory
import math
slim = tf.contrib.slim
metrics = tf.contrib.metrics

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/cifar10',
                    'Directory with the MNIST data.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('eval_interval_secs', 60,
                    'Number of seconds between evaluations.')
flags.DEFINE_integer('num_evals', (10000/32), 'Number of batches to evaluate.')
flags.DEFINE_string('log_dir', './log/eval',
                    'Directory where to log evaluation data.')
flags.DEFINE_string('checkpoint_dir', './log/train',
                    'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS

def load_batch(dataset, batch_size=32, height=32, width=32, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image, label = data_provider.get(['image', 'label'])
    
    preprocessing_fn = preprocessing_factory.get_preprocessing("cifarnet",is_training)
    
    image = preprocessing_fn(
        image,
        height,
        width)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, labels
def main(args):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load the dataset
    dataset = cifar10.get_split('test', FLAGS.data_dir)

    # load batch
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=False)
    print(images,labels)
    # get the model prediction
    network_fn =nets_factory.get_network_fn("cifarnet",num_classes= 10,is_training=False)
    # run the image through the model
#    predictions,_ = lenet.lenet(images)
    predictions,_ = network_fn(images)
    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(predictions, 1))

    # streaming metrics to evaluate
    metrics_to_values, metrics_to_updates = metrics.aggregate_metric_map({
        'mse': metrics.streaming_mean_squared_error(predictions, labels),
        'accuracy': metrics.streaming_accuracy(predictions, labels),
#        'Recall_3': slim.metrics.streaming_recall_at_k(predictions, labels, 3),
    })

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.items():
        summary_name = 'eval/%s' % metric_name
        tf.summary.scalar(summary_name, metric_value)
        
#    for name, value in metrics_to_values.items():
#        summary_name = 'eval/%s' % name
#        op = tf.summary.scalar(summary_name, value, collections=[])
#        op = tf.Print(op, [value], summary_name)
#        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        
        

    # evaluate on the model saved at the checkpoint directory
    # evaluate every eval_interval_secs
#    slim.evaluation.evaluation_loop(
#        '',
#        FLAGS.checkpoint_dir,
#        FLAGS.log_dir,
#        num_evals=FLAGS.num_evals,
#        eval_op=list(metrics_to_updates.values()),
#        eval_interval_secs=FLAGS.eval_interval_secs)

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    num_batches = math.ceil(10000 / float(FLAGS.batch_size))
    metric_values =slim.evaluation.evaluate_once(
        master ='',
        checkpoint_path =checkpoint_path,
        logdir =FLAGS.log_dir,
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        final_op=list(metrics_to_values.values()) 
        )
    for metric, value in zip(metrics_to_values.keys(), metric_values):
     print("%s: %f" %(metric, value))
     
if __name__=='__main__':
    tf.app.run()
