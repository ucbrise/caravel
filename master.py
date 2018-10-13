from subprocess import Popen
from shlex import split
import click
import socket
import time
def find_unbound_port(n=1):
    socks = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        socks.append(s)
    addrs = [s.getsockname()[1] for s in socks]
    [s.close() for s in socks]
    return addrs

@click.command()
@click.option("--mem-frac", type=float, required=True)
@click.option("--allow-growth", is_flag=True, required=True)
@click.option("--result-dir", required=True)
@click.option("--num-procs", "-n", type=int, default=5, required=True)
def master(mem_frac, allow_growth, result_dir, num_procs):
    child_procs = []
    driver_cmd = f'python driver.py --result-dir {result_dir}'
    client_cmd = f'python client.py --mem-frac {mem_frac}'
    if allow_growth:
        client_cmd += ' --allow-growth'
    
    # run driver
    ports = find_unbound_port(num_procs)
    for p in ports:
        port_arg = f' --port {p}'
        driver_cmd += port_arg
    driver_proc = Popen(split(driver_cmd))
    time.sleep(1)

    for p in ports:
        port_arg = f' --port {p}'
        cmd = split(client_cmd + port_arg)
        child_procs.append(Popen(cmd))
        
    # run driver
    driver_proc.wait()
    [p.kill() for p in child_procs]

if __name__ == "__main__":
    master()