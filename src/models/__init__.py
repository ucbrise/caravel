from tensorflow.core.protobuf import config_pb2

SUPPORTED_MODELS = ["res50", "res152", "mobilenet", "mobilenet-224"]


def get_model(
    model_name, powergraph, power_graph_count, batch_size, mem_frac, allow_growth
):
    """Return a lambda closure"""

    from .tf_models import (
        SUPPORTED_MODELS as tf_models,
        load_tf_power_graph,
        get_input,
        load_tf_sess,
    )

    assert model_name in tf_models

    if powergraph:
        sess, img_tensors, predictions = load_tf_power_graph(
            mem_frac, allow_growth, model_name, power_graph_count
        )
        sess_run = lambda: sess.run(predictions)
    else:
        sess, img_tensor, predictions = load_tf_sess(mem_frac, allow_growth, model_name)
        sess_run = lambda: sess.run(predictions)

    return sess_run
