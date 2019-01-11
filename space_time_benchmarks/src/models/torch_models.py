import torch
import torchvision
from threading import Thread
from queue import Queue

torch.backends.cudnn.enabled = False

SUPPORTED_MODELS = ["torch_res50", "torch_res152", "torch_squeezenet"]
MODEL_SHAPES = [1, 3, 224, 224]


def load_torch_model(model_name, batch_size):
    if model_name == "torch_res50":
        model = torchvision.models.resnet50().cuda()
    elif model_name == "torch_res152":
        model = torchvision.models.resnet152().cuda()
    elif model_name == "torch_squeezenet":
        model = torchvision.models.squeezenet1_1().cuda()

    shape = list(MODEL_SHAPES)
    shape[0] = batch_size
    inp = torch.randn(*shape).cuda()

    model(inp)
    torch.cuda.empty_cache()
    return model, inp


def _run_inference_subgraph(inp_queue, out_queue, model_name, batch_size):
    with torch.cuda.stream(torch.cuda.Stream()):
        model, inp = load_torch_model(model_name, batch_size)

        with torch.no_grad():
            model(inp)
        torch.cuda.empty_cache()

        while True:
            inp_queue.get()
            with torch.no_grad():
                model(inp)
            out_queue.put("")


def load_torch_power_graph(model_name, power_graph_count, batch_size=1):
    inp_qs = [Queue() for _ in range(power_graph_count)]
    out_qs = [Queue() for _ in range(power_graph_count)]
    ts = [
        Thread(
            target=_run_inference_subgraph, args=(inp_q, out_q, model_name, batch_size)
        )
        for inp_q, out_q in zip(inp_qs, out_qs)
    ]
    [t.start() for t in ts]
    return inp_qs, out_qs, ts
