import os
from itertools import product
import time

approaches = ["mux", "mps", "batch"]
# models = ["res50", "res152", "mobilenet"]
models = ["torch_res50", "torch_res152", "torch_squeezenet"]
replicas = {
    "torch_res50": list(range(1, 10)),
    "torch_res152": list(range(1, 5)),
    "torch_squeezenet": list(range(1, 15)),
}
# placement_policy = {"mux": [0, 1, 2, 4], "mps": [0, 1, 2, 4], "batch": [0, 1]}
placement_policy = {"mux": [1], "mps": [1], "batch": [1]}
result_dir_root = "learningsys-2018-gpu-mux"
force = [False, True]


def generate_command(approach, model, replica, force, placement_policy):
    mps_req = "start-mps" if approach.startswith("mps") else "stop-mps"

    name = f"{approach}-{model}-{replica}-pp{placement_policy}"
    if force:
        name += "-force"

    cmd = f"""python src/master.py --result-dir \
    {os.path.join(result_dir_root, 'result', approach, model, str(replica), str(placement_policy))} \
    --num-procs {replica} --model-name {model} --placement-policy {placement_policy} \
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
    for approach in approaches:
        for model in models:
            for replica in replicas[model]:
                for pp in placement_policy[approach]:
                    if "res152" in model and approach == "batch":
                        continue
                    all_names.append(
                        generate_command(
                            approach, model, replica, force=False, placement_policy=pp
                        )
                    )

    print(
        f"""
all: {' '.join(all_names)}
    """
    )


if __name__ == "__main__":
    main()
