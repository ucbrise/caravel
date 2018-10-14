import time
from datetime import datetime

import click
import redis
import zmq

import ujson as json
from tf_test import get_input, load_tf_sess


def _block_until(key, val):
    r = redis.Redis()
    r.incr(key)
    while int(r.get(key)) != val:
        pass


@click.command()
@click.option("--port", "-p")
@click.option("--mem-frac", type=float)
@click.option("--allow-growth", is_flag=True)
@click.option("--num-procs", type=int)
def start_client(port, mem_frac, allow_growth, num_procs):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")
    print(f"[Client] Bind to port {port}")

    sess, img_tensor, predictions = load_tf_sess(mem_frac, allow_growth)
    input_img = get_input()

    # Model Warmup
    for _ in range(200):
        pred_prob = sess.run(predictions, feed_dict={img_tensor: input_img})
    print(f"[Client] Warmup finished")

    _block_until("warmup-lock", num_procs)

    query_count = 0
    recent_proc_time = 0.0
    driver_sent_time = ""
    while True:
        data = {
            "query_id": query_count,
            "duration_s": recent_proc_time,
            "sent_time_s": driver_sent_time,
        }
        sock.send_string(json.dumps(data))
        # print('[Client] send, now recv')
        driver_sent_time = sock.recv()

        start = time.perf_counter()
        pred_prob = sess.run(predictions, feed_dict={img_tensor: input_img})
        end = time.perf_counter()
        query_count += 1
        recent_proc_time = end - start


if __name__ == "__main__":
    start_client()
