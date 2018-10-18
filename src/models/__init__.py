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
        sess, img_tensor, predictions = load_tf_sess(
            mem_frac, allow_growth, model_name, batch_size
        )
        sess_run = lambda: sess.run(predictions)

    return sess_run


def get_model_pytorch(
    model_name,
    powergraph,
    power_graph_count,
    batch_size,
    mem_frac=None,
    allow_growth=None,
):
    if powergraph:
        input_queue, output_queue = load_torch_power_graph(
            model_name, power_graph_count
        )

        def run_one_predict():
            for _ in range(power_graph_count):
                input_queue.put("")
            for _ in range(power_graph_count):
                output_queue.get()
