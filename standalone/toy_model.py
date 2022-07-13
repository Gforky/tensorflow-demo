import tensorflow as tf
from tensorflow.python.ops.losses import losses
 
def mlp(input, hidden_sizes):
  hidden_sizes = list(map(int, hidden_sizes.split(",")))
  with tf.name_scope("mlp"):
    for i, hidden_size in enumerate(hidden_sizes):
      # 使用截断正太分布初始化器，创建当前隐层参数
      weights = tf.get_variable(f"weights_{i}",
                                [input.get_shape()[1].value, hidden_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                trainable=True)
      # 训练过程中打印weights值
      # weights = tf.Print(weights, [weights], message=f"\nlayer {i} weights: ", summarize=10)
      # 使用零初始化器创建当前隐层偏置
      bias = tf.zeros(name=f"bias_{i}", shape=[hidden_size])
      # 训练过程中打印bias值
      # bias = tf.Print(bias, [bias], message=f"\nlayer {i} bias: ", summarize=10)
      # y = W*x + b
      output = tf.add(tf.matmul(input, weights), bias)
      # 训练过程中打印layer输出
      # output = tf.Print(output, [output], message=f"\nlayer {i} output: ", summarize=10)
      input = output
  return output

def multiclass_classifier(labels, features, hidden_size, is_train=True):
    logits = mlp(features, hidden_size)
    probabilities = tf.nn.softmax(logits)
    if not is_train:
          return probabilities, None
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    
    return probabilities, loss