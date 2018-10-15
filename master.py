import socket
import time
from shlex import split
from subprocess import Popen

import click
import redis

from tf_util import SUPPORTED_MODELS


def find_unbound_port(n=1):
    socks = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        socks.append(s)
    addrs = [s.getsockname()[1] for s in socks]
    [s.close() for s in socks]
    return addrs


@click.command()
@click.option("--mem-frac", type=float, required=True)
@click.option("--allow-growth", is_flag=True, required=True)
@click.option("--result-dir", required=True)
@click.option("--num-procs", "-n", type=int, default=5, required=True)
@click.option("--model-name", type=click.Choice(SUPPORTED_MODELS), required=True)
@click.option("--power-graph", is_flag=True)
@click.option("--force", is_flag=True)
def master(
    mem_frac, allow_growth, result_dir, num_procs, model_name, power_graph, force
):
    # reset the warmup lock
    r = redis.Redis()
    r.set("warmup-lock", 0)
    r.set("connect-lock", 0)

    child_procs = []
    driver_cmd = f"python driver.py --result-dir {result_dir}"
    client_cmd = f"python client.py --mem-frac {mem_frac} --num-procs {num_procs} --model-name {model_name}"
    if allow_growth:
        client_cmd += " --allow-growth"

    # run driver
    if power_graph:
        ports = find_unbound_port(1)
    else:
        ports = find_unbound_port(num_procs)

    for p in ports:
        port_arg = f" --port {p}"
        driver_cmd += port_arg
    driver_cmd = f"numactl -C {num_procs+1} " + driver_cmd
    if force:
        driver_cmd += " --force"
    driver_proc = Popen(split(driver_cmd))
    time.sleep(1)

    for i, p in enumerate(ports):
        port_arg = f" --port {p}"
        cmd = split(f"numactl -C {i+1} " + client_cmd + port_arg)
        if power_graph:
            cmd += ["--power-graph"]
        child_procs.append(Popen(cmd))
        time.sleep(1)

    # run driver
    driver_proc.wait()
    [p.terminate() for p in child_procs]
    time.sleep(2)


if __name__ == "__main__":
    master()
