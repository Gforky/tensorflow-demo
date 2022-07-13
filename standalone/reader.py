import tensorflow as tf

def parse_func(sample):
    sample = tf.strings.strip(sample)
    feat_vals = tf.string_split(sample, delimiter=",")
    feat_vals = tf.sparse.to_dense(feat_vals)
    labels = tf.slice(feat_vals, [0, 0], [-1, 3])
    features = tf.slice(feat_vals, [0, 3], [-1, -1])
    labels = tf.string_to_number(labels, out_type=tf.float32)
    features = tf.string_to_number(features, out_type=tf.float32)
    features = tf.reshape(features, shape=[-1, 4])
    return labels, features

def get_csv_dataset(data_files, batch_size=8, epoch_num=1):
    dataset = tf.data.Dataset.list_files(data_files)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
                                tf.data.TextLineDataset,
                                cycle_length=15,
                                sloppy=True)
                            )
    dataset = dataset.repeat(epoch_num) # repeat forever
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_func=parse_func, num_parallel_calls=2)
    dataset = dataset.prefetch(1)
    return dataset
