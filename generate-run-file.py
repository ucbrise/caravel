import os
from itertools import product

approaches = ["mux", "mps", "powergraph"]
models = ["res50", "res152", "mobilenet"]
replicas = list(range(1, 6))
result_dir_root = "learning-2018-gpu-mux"


def generate_command(approach, model, replica):
    mps_req = "start-mps" if approach == "mps" else "stop-mps"
    mem_frac = 1 / replica * 0.9
    name = f"{approach}-{model}-{replica}"
    print(
        f"""
{name}: {mps_req}
\t  time python master.py \
        --mem-frac {mem_frac} \
        --allow-growth \
        --result-dir {os.path.join(result_dir_root, 'result', approach, model, str(replica))} \
        --num-procs {replica} \
        --model-name {model} \
        {"--power-graph" if model == "powergraph" else "#"}
    """
    )
    return name


def main():
    print(
        """
start-mps:
\t  bash bin/start-mps.sh
stop-mps:
\t  bash bin/stop-mps.sh
    """
    )

    all_names = []
    for approach, model, replica in product(approaches, models, replicas):
        all_names.append(generate_command(approach, model, replica))

    print(
        f"""
all: {' '.join(all_names)}
    """
    )


if __name__ == "__main__":
    main()
