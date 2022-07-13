from toy_model import multiclass_classifier
import tensorflow as tf
from reader import get_csv_dataset

flags = tf.app.flags
flags.DEFINE_string("hidden_size", None, 
                    "MLP hidden layers' sizes")
flags.DEFINE_integer("batch_size", 8,
                    "batch size while reading data for training")
flags.DEFINE_integer("epoch_num", 10,
                    "epoch number of training")
flags.DEFINE_integer("max_steps", 1000,
                    "max steps of training")
flags.DEFINE_float("learning_rate", 1e-5,
                    "optimizer's learning rate")
flags.DEFINE_string("data_files", None,
                    "data filenames, seperated by comma")
flags.DEFINE_string("model_dir", "./model_dir",
                    "directory for saving model's checkpoints")
FLAGS = flags.FLAGS

def main(_):
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    epoch_num = FLAGS.epoch_num
    max_steps = FLAGS.max_steps
    lr = FLAGS.learning_rate
    data_files = FLAGS.data_files.split(",")
    model_dir = FLAGS.model_dir

    # create data reader
    ds = get_csv_dataset(data_files, batch_size, epoch_num)
    iterator = tf.data.make_initializable_iterator(ds)
    labels, features = iterator.get_next()
    # create classifier
    prob, loss = multiclass_classifier(labels, features, hidden_size)
    # create optimizer
    global_step = tf.train.get_or_create_global_step()
    adam = tf.train.AdamOptimizer(lr)
    batch_grads = adam.compute_gradients(loss)
    train_op = adam.apply_gradients(batch_grads, tf.train.get_global_step())
    # create saver
    saver = tf.train.Saver()
    # create metrics
    # use one-hot prediction result for accuracy calculation
    true_pred = tf.where(
        tf.equal(tf.reduce_max(prob, axis=1, keep_dims=True), prob), 
        tf.ones_like(prob), 
        tf.zeros_like(prob)
    )
    acc, acc_op = tf.metrics.accuracy(labels, true_pred)
    # mean loss
    avg_loss, avg_loss_op = tf.metrics.mean(loss)
    cur_step = 0
    with tf.Session() as sess:
        # initialization
        sess.run(iterator.initializer)
        sess.run(tf.variables_initializer(tf.global_variables()))
        sess.run(tf.variables_initializer(tf.local_variables()))
        # training loop
        while cur_step < max_steps:
            sess.run([train_op, acc_op, avg_loss_op])
            cur_loss, cur_acc, cur_step = sess.run([avg_loss, acc, global_step])
            if cur_step % 100 == 0:
                print("*****batch_loss: ", cur_loss)
                print("*****accuracy: ", cur_acc)
                print("*****gloabl_step: ", cur_step)
                # save checkpoint
                saver.save(sess, f"{model_dir}/model", global_step=cur_step)

if __name__ == "__main__":
    tf.app.run()