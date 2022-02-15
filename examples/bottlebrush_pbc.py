import os
import argparse
import pprint

import numpy as np

import wiggin
import wiggin_mito

from simtk import openmm

assert openmm.Platform_getNumPlatforms() > 3
for i in range(openmm.Platform_getNumPlatforms()):
    print(openmm.Platform_getPlatform(i).getName())


parser = argparse.ArgumentParser()


parser.add_argument(
    "--loop_kb", type=float, default=100, help="The average size of loops in kb."
)

parser.add_argument(
    "--loop_spacing",
    type=int,
    default=1,
    help="The size of a linker between two consecutive loops.",
)

parser.add_argument(
    "--loop_spacing_distr",
    type=str,
    default='uniform',
    help="The distribution of loop sizes (uniform/exp)",
)

parser.add_argument(
    "--loop_n", type=int, default=500, help="The number of loops in the system."
)

parser.add_argument(
    "--rep_e",
    type=float,
    default=1.5,
    help="The maximal energy of repulsion between particles.",
)

parser.add_argument(
    "--axial_density",
    type=float,
    default=25,
    help="The axial density of chromatin in Mb/micron.",
)

parser.add_argument(
    "--replicate", 
    type=int, 
    default=0, 
    help="The replicate index.")

parser.add_argument(
    "--num_blocks", type=int, default=5000, help="The number of blocks to simulate."
)

parser.add_argument(
    "--root_loop_spacers",
    action="store_true",
    help="If provided, the spacers between root loops will be straight.",
)


parser.add_argument(
    "--PBCNuclD", 
    type=float, 
    default=1.25, 
    help="The linear dimension of the PBC box per nucleosome (V=N*PBCNuclDˆ3)."
)

parser.add_argument(
    "--out_folder",
    type=str,
    default="./",
    help="The root folder where the results will be stored.",
)

args = parser.parse_args()

bp_particle = 200
loop_size = int(np.round(args.loop_kb * 1000.0 / bp_particle))
loop_spacing = args.loop_spacing

loop_n = args.loop_n
rep_e = args.rep_e

out_folder = args.out_folder

replicate = args.replicate

num_blocks = args.num_blocks

root_loop_spacers = args.root_loop_spacers

colrate = 0.01
errortol = 0.003

wiggle_dist = 0.25


n = loop_n * loop_size
V = (args.PBCNuclD ** 3) * n

if not(args.PBCNuclD):
    pbcbox = False

elif args.axial_density:
    n_mb = loop_n * loop_size * bp_particle / 1e6
    axial_length_final = n_mb / args.axial_density * 100
    pbcbox_side = (V / axial_length_final) ** (0.5)
    pbcbox = (pbcbox_side, pbcbox_side, axial_length_final)


else:
    pbcbox_size = V ** (1/3)
    pbcbox = (pbcbox_size, pbcbox_size, pbcbox_size)

dir_name_dict = dict(
    Loop=args.loop_kb,
    Spacing=args.loop_spacing,
    SpacingDistr=args.loop_spacing_distr,
    AxDensity=args.axial_density,
    ColRate=colrate,
    ErrorTol=errortol,
    LoopN=args.loop_n,
    RepE=args.rep_e,
    PBCNuclD=args.PBCNuclD,   
    R=replicate,
)

if root_loop_spacers:
    dir_name_dict["RootLoopSpacers"] = 1

name = "-".join(f"{n}_{v}" for n, v in dir_name_dict.items())

c = wiggin.core.SimConstructor(name=name, folder=os.path.join(out_folder, name))


c.add_action(
    wiggin.actions.sim.InitializeSimulation(
        N=loop_size*loop_n,
        # platform='CPU'
        # GPU='1',
        PBCbox=pbcbox,
        error_tol=errortol,
        collision_rate=colrate,
    )
)

c.add_action(
    wiggin.actions.interactions.Chains(
        wiggle_dist=wiggle_dist,
        repulsion_e=rep_e),
)

c.add_action(
    wiggin_mito.actions.loops.SingleLayerLoopPositions(
        loop_size=loop_size,
        loop_spacing=loop_spacing,
        loop_spacing_distr=args.loop_spacing_distr,
        ),
)



c.add_action(
    wiggin_mito.actions.interactions.HarmonicLoops(
        wiggle_dist=wiggle_dist,
    ),
)


if root_loop_spacers:
    c.add_action(
        wiggin_mito.actions.interactions.RootLoopSeparator(),
    )

if args.axial_density:
    c.add_action(
        wiggin_mito.actions.conformations.RWLoopBrushConformation(end=(0, 0, axial_length_final)),
    )

    c.add_action(
        wiggin_mito.actions.constraints.TetherTips(
            k = (2, 2, 2),
            particles = (0, -1),
            positions = [(0, 0, 0), (0, 0, axial_length_final)]

        )
    )

else:
    c.add_action(
        wiggin_mito.actions.conformations.RWLoopBrushConformation(),
    )


c.add_action(
    wiggin.actions.sim.LocalEnergyMinimization()
)

c.add_action(
    wiggin.actions.sim.BlockStep(
        num_blocks=num_blocks,
    ),
)


pprint.pprint(c.action_args)

c.configure()

pprint.pprint(c.shared_config)
pprint.pprint(c.action_configs)

c.save_config()

c.run_init()

c.run_loop()
