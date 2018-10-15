import os
from itertools import product

approaches = ["mux", "mps", "powergraph"]
models = ["res50", "res152", "mobilenet"]
replicas = list(range(1, 6))
result_dir_root = "learningsys-2018-gpu-mux"


def generate_command(approach, model, replica):
    mps_req = "start-mps" if approach == "mps" else "stop-mps"
    mem_frac = 1 / replica * 0.9
    name = f"{approach}-{model}-{replica}"
    print(
        f"""
{name}:
\t  bash bin/{mps_req}.sh
\t  python master.py \
        --mem-frac {mem_frac} \
        --result-dir {os.path.join(result_dir_root, 'result', approach, model, str(replica))} \
        --num-procs {replica} \
        --model-name {model} \
        {"--power-graph" if model == "powergraph" else "#"}
    """
    )
    return name


def main():
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
