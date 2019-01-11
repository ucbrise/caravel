SUPPORTED_MODELS = ["res50", "res152", "mobilenet", "mobilenet-224"]
SUPPORTED_MODELS += ["torch_res50", "torch_res152", "torch_squeezenet"]


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
    from .torch_models import load_torch_model, load_torch_power_graph
    import torch

    torch.backends.cudnn.enabled = False

    if powergraph and batch_size == 1:
        inp_qs, out_qs, ts = load_torch_power_graph(model_name, power_graph_count)

        def run_one_predict():
            [inp_q.put("") for inp_q in inp_qs]
            [out_q.get() for out_q in out_qs]

        return run_one_predict, ts
    else:
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            model, inp = load_torch_model(model_name, batch_size)

        def run_one_predict():
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    model(inp)

        return run_one_predict, None
