import os
from itertools import product

# approaches = ["mux", "mps", "batch"]
approaches = ["mps"]
# models = ["res50", "res152", "mobilenet", "mobilenet-224"]
# models = ["mobilenet", "res50"]
models = ["mobilenet"]
replicas = list(range(10, 40))
placement_policy = [0]
result_dir_root = "learningsys-2018-gpu-mux/p3-8xlarge-random-placement"
force = [False, True]


def generate_command(approach, model, replica, force, placement_policy):
    mps_req = "start-mps" if approach.startswith("mps") else "stop-mps"

    name = f"{approach}-{model}-{replica}"
    if force:
        name += "-force"

    cmd = f"""python src/master.py --result-dir \
    {os.path.join(result_dir_root, 'result', approach, model, str(replica))} \
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
    for approach, model, replica, pp in product(
        approaches, models, replicas, placement_policy
    ):
        # if approach == "mux" and model == "res50" and replica > 10:
        # if replica > 11:
        #     continue
        all_names.append(
            generate_command(approach, model, replica, force=False, placement_policy=pp)
        )

    print(
        f"""
all: {' '.join(all_names)}
    """
    )

    all_names = []
    for approach, model, replica, pp in product(
        approaches, models, replicas, placement_policy
    ):
        all_names.append(
            generate_command(approach, model, replica, force=True, placement_policy=pp)
        )
    print(
        f"""
all-force: {' '.join(all_names)}
    """
    )


if __name__ == "__main__":
    main()
