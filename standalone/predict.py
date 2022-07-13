import tensorflow as tf
import numpy as np

def gen_mock_input(meta_graph):
    mock_input = {}
    input_def = meta_graph.signature_def["serving_default"].inputs
    model_inputs = []
    for key, val in input_def.items():
        model_inputs.append((key, val))
    for feat_name, feat_spec in model_inputs:
        var_shape = list(map(lambda x: x.size, feat_spec.tensor_shape.dim))
        var_shape = list(map(lambda x: x if x > 0 else 1, var_shape))
        print(f"{feat_name}: {feat_spec.dtype}, {var_shape}")
        mock_input[f"{feat_name}:0"] = np.ones(shape=var_shape, dtype=float)
    return mock_input

def get_output_names(meta_graph):
    output_def = list(meta_graph.signature_def["serving_default"].outputs.values())
    output_def = [x.name for x in output_def]
    return output_def

with tf.Session() as sess:
    meta_graph = tf.saved_model.load(sess, ["serve"], "iris_model/savedmodel")
    output_names = get_output_names(meta_graph)
    pred_result = sess.run(output_names, feed_dict=gen_mock_input(meta_graph))
    print(pred_result)
