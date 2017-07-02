************************************
* A simple working training script *
************************************
  # Load data and create the model:
  images, labels = LoadData(...)
  predictions = MyModel(images)
  # Define the loss:
  slim.losses.log_loss(predictions, labels)
  total_loss = slim.losses.get_total_loss()
  # Define the optimizer:
  optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  # Run training.
  slim.learning.train(train_op, my_log_dir)
*************************
* Creating the train_op *
*************************
In order to train, TF-Slim's train loop needs a train_op: an `Operation` that
(a) computes the loss, (b) applies the gradients to update the weights and
(c) returns the value of the loss. slim.learning.create_train_op creates
such an `Operation`. This function also provides the ability to manipulate
the gradients using a few arguments:
  # Create the train_op and clip the gradient norms:
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      clip_gradient_norm=4)
  # Create the train_op and scale the gradients by providing a map from variable
  # name (or variable) to a scaling coefficient:
  gradient_multipliers = {
    'conv0/weights': 1.2,
    'fc8/weights': 3.4,
  }
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      gradient_multipliers=gradient_multipliers)
****************************************************************
* Performing additional (non-gradient) updates during training *
****************************************************************
Many networks utilize modules, like BatchNorm, that require performing a series
of non-gradient updates during training. slim.learning.create_train_op allows
a user to pass in a list of update_ops to call along with the gradient updates.
  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops)
By default, slim.learning.create_train_op includes all update ops that are
part of the `tf.GraphKeys.UPDATE_OPS` collection. Additionally, TF-Slim's
slim.batch_norm function adds the moving mean and moving variance updates to
this collection. Consequently, users who want to use slim.batch_norm will not
need to take any additional steps in order to have the moving mean and moving
variance updates be computed.
However, users with additional, specialized updates can either override the
default update ops or simply add additional update ops to the
`tf.GraphKeys.UPDATE_OPS` collection:
  # Force TF-Slim NOT to use ANY update_ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=[])
  # Use an alternative set of update ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=my_other_update_ops)
  # Use an alternative set of update ops in addition to the default updates:
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update0)
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update1)
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer)
  # Which is the same as:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
******************************************
* Initializing a model from a checkpoint *
******************************************
It is common to want to 'warm-start' a model from a pre-trained checkpoint.
TF-Slim provides a convenient mechanism for doing so:
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  # Create the initial assignment op
  checkpoint_path = '/path/to/old_model_checkpoint'
  variables_to_restore = slim.get_model_variables()
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
***************************************************************************
* Initializing a model from a checkpoint whose variable names don't match *
***************************************************************************
At times, a user may want to initialize a new model with values from a
checkpoint whose variable names do not match those of the current model. In this
case, one needs to create a mapping from the checkpoint variable names to the
current model variables. This requires only a small modification of the code
above:
  ...
  # Creates a model with two variables, var0 and var1
  predictions = MyModel(images)
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  checkpoint_path = '/path/to/old_model_checkpoint'
  # Create the mapping:
  variables_to_restore = {
      'name_var_0_in_checkpoint': slim.get_unique_variable('var0'),
      'name_var_1_in_checkpoint': slim.get_unique_variable('var1')
  }
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
*************************************************
* Fine-Tuning Part of a model from a checkpoint *
*************************************************
Rather than initializing all of the weights of a given model, we sometimes
only want to restore some of the weights from a checkpoint. To do this, one
need only filter those variables to initialize as follows:
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  checkpoint_path = '/path/to/old_model_checkpoint'
  # Specify the variables to restore via a list of inclusion or exclusion
  # patterns:
  variables_to_restore = slim.get_variables_to_restore(
      include=["conv"], exclude=["fc8", "fc9])
  # or
  variables_to_restore = slim.get_variables_to_restore(exclude=["conv"])
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
******************************************************
* Initializing model variables from values in memory *
******************************************************
One may want to initialize the weights of a model from values from an arbitrary
source (a text document, matlab file, etc). While this is technically feasible
using plain TensorFlow, it also results in the values of your weights being
stored in the graph. For large models, this becomes prohibitively large. TF-Slim
allows you to perform this initial assignment without having to store the values
of the initial model in the graph itself by using placeholders and a feed
dictionary:
  ...
  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)
  # Create the mapping from variable names to values:
  var0_initial_value = ReadFromDisk(...)
  var1_initial_value = ReadFromDisk(...)
  var_names_to_values = {
    'var0': var0_initial_value,
    'var1': var1_initial_value,
  }
  init_assign_op, init_feed_dict = slim.assign_from_values(var_names_to_values)
  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)
  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
"""