import tensorflow as tf
from tensorflow.python.training import checkpoint_management
from toy_model import multiclass_classifier

flags = tf.app.flags
flags.DEFINE_string("hidden_size", None, 
                    "MLP hidden layers' sizes")
flags.DEFINE_string("model_dir", "./model_dir",
                    "directory for saving model's checkpoints")
FLAGS = flags.FLAGS

def main(_):
    hidden_size = FLAGS.hidden_size
    model_dir = FLAGS.model_dir
    savedmodel_input = tf.placeholder(tf.float32, [None, 4], name="features")
    prob, _ = multiclass_classifier(None, savedmodel_input, hidden_size, is_train=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        lastest_ckpt = checkpoint_management.latest_checkpoint(model_dir)
        saver.restore(sess, lastest_ckpt)
        tf.saved_model.simple_save(sess, f"{model_dir}/savedmodel", {"features": savedmodel_input}, {"probabilities": prob})

if __name__ == "__main__":
    tf.app.run()