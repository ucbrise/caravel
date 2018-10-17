import os
from itertools import product

approaches = ["mux", "mps", "batch"]
# models = ["res50", "res152", "mobilenet", "mobilenet-224"]
models = ["mobilenet", "res50"]
replicas = list(range(1, 15))
result_dir_root = "learningsys-2018-gpu-mux/p3-8xlarge"
force = [False, True]


def generate_command(approach, model, replica, force):
    mps_req = "start-mps" if approach.startswith("mps") else "stop-mps"

    name = f"{approach}-{model}-{replica}"
    if force:
        name += "-force"

    cmd = f"""python src/master.py --result-dir \
    {os.path.join(result_dir_root, 'result', approach, model, str(replica))} \
    --num-procs {replica} --model-name {model} \
    """
    if approach == "powergraph":
        cmd += "\t --power-graph"
    if approach == "batch":
        cmd += "\t --batch"
    if force:
        cmd += "\t --force"

    print(
        f"""
{name}:
\t  bash bin/{mps_req}.sh
\t  {cmd}
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
