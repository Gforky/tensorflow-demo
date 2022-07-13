import tensorflow as tf
from reader import get_csv_dataset

sess = tf.Session()
ds = get_csv_dataset("iris_train.csv", 1)
iterator = tf.data.make_initializable_iterator(ds)
labels, features = iterator.get_next()
for _ in range(10):
  sess.run(iterator.initializer)
  sess.run([labels, features])
