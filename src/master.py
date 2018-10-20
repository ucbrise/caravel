import os
import socket
import time
from shlex import split
from subprocess import Popen
import click
import redis
import numpy as np

TOTAL_CORES = 32


class ClientRun:
    def __init__(self, model_name, result_path, num_procs):
        self.model_name = model_name
        self.result_path = result_path
        self.num_proc = num_procs

        self.mem_frac = str(0.95 / self.num_proc)
        self.allow_growth = False

        self.power_graph = False
        self.batch_size = 1
        self.force = False

        self.proc = None

        self.core = None

    def run(self):
        cmd = ["python", "src/client.py"]
        cmd += ["--model-name", self.model_name]
        cmd += ["--result-path", self.result_path]
        cmd += ["--num-replicas", self.num_proc]
        cmd += ["--mem-frac", self.mem_frac]
        if self.allow_growth:
            cmd += ["--allow-growth"]
        if self.power_graph:
            cmd += ["--power-graph", "--power-graph-count", self.power_graph_count]
        if self.batch_size != 1:
            cmd += ["--batch-size", self.batch_size]
        if self.force:
            cmd += ["--force"]

        if self.core is not None:
            cmd = ["numactl", "-C", self.core] + cmd

        cmd = split(" ".join(map(str, cmd)))
        print(" ".join(cmd))
        self.proc = Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))

    def wait(self):
        self.proc.wait()

    def set_tf_mem_frac(self, frac):
        self.mem_frac = str(frac)

    def set_tf_allow_growth(self):
        self.allow_growth = True

    def set_power_graph(self, n):
        self.power_graph = True
        self.power_graph_count = n

    def set_batch_size(self, n):
        self.batch_size = n

    def set_force(self):
        self.force = True

    def set_core(self, core):
        self.core = core

    def set_num_proc(self, num_proc):
        self.num_proc = num_proc


@click.command()
@click.option("--mem-frac", type=float)
@click.option("--allow-growth", is_flag=True)
@click.option("--result-dir", required=True)
@click.option("--num-procs", "-n", type=int, default=5, required=True)
@click.option("--model-name", required=True)
@click.option("--power-graph", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--batch", is_flag=True)
@click.option("--placement-policy", type=int, default=1)
def master(
    mem_frac,
    allow_growth,
    result_dir,
    num_procs,
    model_name,
    power_graph,
    force,
    batch,
    placement_policy,
):

    # reset the warmup lock
    r = redis.Redis()
    r.set("warmup-lock", 0)
    r.set("connect-lock", 0)

    clients = [
        ClientRun(model_name, os.path.join(result_dir, f"{i}.pq"), num_procs)
        for i in range(num_procs)
    ]
    if mem_frac:
        [c.set_tf_mem_frac(mem_frac) for c in clients]

    if allow_growth:
        [c.set_tf_allow_growth() for c in clients]

    if batch:
        batch_client = clients[0]
        batch_client.set_batch_size(num_procs)
        batch_client.set_num_proc(1)
        clients = [batch_client]

    if force:
        [c.set_force() for c in clients]

    if power_graph:
        power_graph_client = clients[0]
        power_graph_client.set_power_graph(num_procs)
        clients = [power_graph_client]

    if placement_policy != 0:  # not random placement
        if TOTAL_CORES * placement_policy < len(clients):
            raise Exception(
                f"We have {num_procs} clients but we can only fit {TOTAL_CORES*placement}. Please change the placement policy"
            )
        all_cores = np.arange(TOTAL_CORES)
        expanded = np.repeat(all_cores, int(placement_policy))
        for core, client in zip(expanded, clients):
            client.set_core(core)

    [c.run() for c in clients]
    [c.wait() for c in clients]


if __name__ == "__main__":
    master()
