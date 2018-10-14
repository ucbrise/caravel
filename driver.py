import click
import zmq
import ujson as json
import os
import time
import pandas as pd


@click.command()
@click.option("--port", "-p", multiple=True)
@click.option("--result-dir")
def send_queries(port, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    ctx = zmq.Context()
    poller = zmq.Poller()

    sockets = {}
    for i, p in enumerate(port):
        socket = ctx.socket(zmq.REP)
        socket.bind(f"tcp://127.0.0.1:{p}")
        print(f"[Driver] Bind to {p}")
        poller.register(socket, zmq.POLLIN)
        sockets[socket] = i

    results = {i: [] for i in range(len(port))}
    while True:
        ready_socks = dict(poller.poll())
        for s in ready_socks:
            msg = json.loads(s.recv())
            # print(f"[Driver] recved {msg}")

            # handle handshake msg
            if msg["query_id"] == 0:
                s.send_string(str(time.time()))
                continue

            msg["recv_time_s"] = time.time()
            results[sockets[s]].append(msg)
            s.send_string(str(time.time()))

        if all([len(lst) > 1000 for lst in results.values()]):
            break

    for i, lst in results.items():
        df = pd.DataFrame.from_dict(lst)
        df.to_parquet(f"{result_dir}/{i}.pq")
        print(df["duration_s"].mean())


if __name__ == "__main__":
    send_queries()
