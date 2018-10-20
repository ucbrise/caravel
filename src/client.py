import time
import click
import redis
import os
import pandas as pd
import models
import numpy as np


def _block_until(key, val):
    """A semaphore implemented in redis"""
    r = redis.Redis()
    r.incr(key)
    while int(r.get(key)) != val:
        pass


def _log(*args, **kwargs):
    print("[Client]", *args, **kwargs)


@click.command()
@click.option("--mem-frac", type=float, required=True)
@click.option("--allow-growth", is_flag=True)
@click.option("--num-replicas", type=int, required=True)
@click.option("--model-name", type=click.Choice(models.SUPPORTED_MODELS), required=True)
@click.option("--power-graph", is_flag=True)
@click.option("--power-graph-count", type=int, default=1)
@click.option("--batch-size", type=int, default=1)
@click.option("--result-path", required=True)
@click.option("--force", is_flag=True)
def start_client(
    mem_frac,
    allow_growth,
    num_replicas,
    model_name,
    power_graph,
    power_graph_count,
    batch_size,
    result_path,
    force,
):
    if os.path.exists(result_path) and not force:
        _log(f"Path {result_path} exists. Skipping")
        return
    os.makedirs(os.path.split(result_path)[0], exist_ok=True)

    require_locks = not (power_graph or batch_size != 1)
    print(f"Require Locks {require_locks}")
    # Load Model
    if model_name.startswith("torch_"):
        sess_run, threads = models.get_model_pytorch(
            model_name, power_graph, power_graph_count, batch_size
        )
    else:
        sess_run = models.get_model(
            model_name,
            power_graph,
            power_graph_count,
            batch_size,
            mem_frac,
            allow_growth,
        )

    if require_locks:
        _block_until("connect-lock", num_replicas)
    _log("Model Loaded")

    # Model Warmup
    for _ in range(200):
        sess_run()
    _log("Warmup finished")
    if require_locks:
        _block_until("warmup-lock", num_replicas)

    # Model Evaluation
    durations = []
    for _ in range(2000):
        start = time.perf_counter()
        sess_run()
        end = time.perf_counter()
        duration_ms = (end - start) * 1000
        durations.append(duration_ms)

    durations = durations[500:1500]

    if require_locks:
        _block_until("exit-lock", num_replicas)
    # Save Data
    df = pd.DataFrame({"duration_ms": durations})
    df.to_csv(result_path)
    mean, p99 = df["duration_ms"].mean(), np.percentile(durations, 99)
    _log(f"Mean Latency: {mean}, P99: {p99}")
    import sys

    sys.exit(0)


if __name__ == "__main__":
    start_client()
