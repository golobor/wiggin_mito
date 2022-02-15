#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00 
#SBATCH --partition=g
#SBATCH --gres=gpu:1
#SBATCH --qos=g_medium
#SBATCH --constraint=g3
#SBATCH --job-name=sweep_pbc
#SBATCH --output=./logs/mitosweep_40_pbc-%A_%a.out
#SBATCH --error=./logs/mitosweep_40_pbc-%A_%a.err

import subprocess
import os
import numpy as np
import itertools
import logging

logging.basicConfig(level=logging.INFO)

cli_base_params = {
    "--rep_e": 1.5,
    "--loop_n": 500,
    "--loop_spacing": 3,
    "--loop_kb": 100,
    "--replicate": 0,
    "--num_blocks": 5000,
    "--root_loop_spacers": "",
    '--axial_density' : 0,
    '--out_folder' : './'
}


cli_grids = [

    {
        '--rep_e':[1.5],
        "--loop_spacing_distr": ['uniform'],
        '--loop_spacing': [1],
        '--axial_density': [10, 20, 30],
        "--replicate": [1],
    },

]


N_PARAM_COMBOS = sum(np.prod([len(v) for v in grid.values()]) for grid in cli_grids)

logging.info(f"Sweeping over {N_PARAM_COMBOS} parameter combinations!")

dry_run = True if "SLURM_ARRAY_TASK_ID" not in os.environ else False
if dry_run:
    logging.warning("SLURM_ARRAY_TASK_ID not found, executing a dry run!")
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
logging.info(f"Executing task # {task_id}")


cli_extra_params = next(
    itertools.islice(
        itertools.chain.from_iterable(
            itertools.product(*[[(k, val) for val in vals] for k, vals in grid.items()])
            for grid in cli_grids
        ),
        task_id,
        task_id + 1,
    )
)
print(cli_extra_params)
cli_extra_params = dict(cli_extra_params)

cli_params = dict(cli_base_params)
cli_params.update(cli_extra_params)

cmd = [
    "python",
    "./bottlebrush_pbc.py",
]

for k, v in cli_params.items():
    if v is not None:
        cmd.append(k)
        if bool(str(v)):
            cmd.append(str(v))

logging.info(f"Executing command: {repr(cmd)}")

if not dry_run:
    subprocess.call(cmd)
