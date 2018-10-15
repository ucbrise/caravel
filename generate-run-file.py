import os
from itertools import product

# approaches = ["mux", "mps", "powergraph", "mps-even", "mps-even-times-2", "batch"]
approaches = ["mux", "mps", "powergraph", "batch"]
# models = ["res50", "res152", "mobilenet", "mobilenet-224"]
models = ["mobilenet-224"]
replicas = list(range(1, 6))
result_dir_root = "learningsys-2018-gpu-mux"
force = [False, True]


def generate_command(approach, model, replica, force):
    mps_req = "start-mps" if approach.startswith("mps") else "stop-mps"
    mem_frac = 0.1
    name = f"{approach}-{model}-{replica}"
    if force:
        name += "-force"

    mps_strategy = "default"
    if approach == "mps-even":
        mps_strategy = "even"
    elif approach == "mps-even-times-2":
        mps_strategy = "even_times_2"

    print(
        f"""
{name}:
\t  bash bin/{mps_req}.sh
\t  python master.py \
        --mem-frac {mem_frac} \
        --allow-growth \
        --result-dir {os.path.join(result_dir_root, 'result', approach, model, str(replica))} \
        --num-procs {replica} \
        --model-name {model} \
        --mps-thread-strategy {mps_strategy} \
        {"--power-graph" if approach in ["powergraph", "batch"] else "#"} \
        {"--batch" if approach == "batch" else "#"} \
        {'--force' if force else '#'}
    """
    )
    return name


def main():
    all_names = []
    for approach, model, replica in product(approaches, models, replicas):
        all_names.append(generate_command(approach, model, replica, force=False))

    print(
        f"""
all: {' '.join(all_names)}
    """
    )

    all_names = []
    for approach, model, replica in product(approaches, models, replicas):
        all_names.append(generate_command(approach, model, replica, force=True))
    print(
        f"""
all-force: {' '.join(all_names)}
    """
    )


if __name__ == "__main__":
    main()
