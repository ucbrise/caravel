import time
from datetime import datetime

import click
import redis
import zmq

import ujson as json
from tf_util import SUPPORTED_MODELS, get_input, load_tf_power_graph, load_tf_sess


def _block_until(key, val):
    r = redis.Redis()
    r.incr(key)
    while int(r.get(key)) != val:
        pass


@click.command()
@click.option("--port", "-p", required=True)
@click.option("--mem-frac", type=float, required=True)
@click.option("--allow-growth", is_flag=True)
@click.option("--num-procs", type=int, required=True)
@click.option("--model-name", type=click.Choice(SUPPORTED_MODELS), required=True)
@click.option("--power-graph", is_flag=True)
def start_client(port, mem_frac, allow_growth, num_procs, model_name, power_graph):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")
    print(f"[Client] Bind to port {port}")

    if power_graph:
        print(f"[Client] Coalescing {num_procs} graphs into one graph!")
        sess, img_tensors, predictions = load_tf_power_graph(
            mem_frac, allow_growth, model_name, num_procs
        )
        input_img = get_input(model_name)
        sess_run = lambda: sess.run(
            predictions, feed_dict={img_tensor: input_img for img_tensor in img_tensors}
        )
    else:
        sess, img_tensor, predictions = load_tf_sess(mem_frac, allow_growth, model_name)
        input_img = get_input(model_name)
        sess_run = lambda: sess.run(predictions, feed_dict={img_tensor: input_img})

    # Model Warmup
    for _ in range(200):
        sess_run()
    print(f"[Client] Warmup finished")

    if not power_graph:
        _block_until("warmup-lock", num_procs)

    query_count = 0
    recent_proc_time_ms = 0.0
    driver_sent_time = ""
    while True:
        data = {
            "query_id": query_count,
            "duration_ms": recent_proc_time_ms,
            "sent_time_ms": driver_sent_time,
        }
        sock.send_string(json.dumps(data))
        driver_sent_time = sock.recv()

        start = time.perf_counter()
        sess_run()
        end = time.perf_counter()
        query_count += 1
        recent_proc_time_ms = (end - start) * 1000


if __name__ == "__main__":
    start_client()
