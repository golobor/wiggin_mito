import pprint

import wiggin
import wiggin_mito

c = wiggin.core.SimConstructor()

c.add_action(
    wiggin.actions.sim.InitializeSimulation(
        N=400*5*500,
        # platform='CPU'
        # GPU='1',
        error_tol=0.01,
        collision_rate=0.003,
    ),
)

c.add_action(
    wiggin.actions.interactions.Chains(
        wiggle_dist=0.25,
        repulsion_e=1.5),
)

c.add_action(
    wiggin_mito.actions.loops.TwoLayerLoopPositions(
        inner_loop_size=400,
        outer_loop_size=400 * 5,
        ),
)

c.add_action(
    wiggin_mito.actions.conformations.UniformHelicalLoopBrushConformation(
        period_particles=int(8e6 // (200 * 400 * 5)) * 2,
        helix_step=15,
        axial_compression_factor=4.0
    )
)


c.add_action(
    wiggin_mito.actions.interactions.HarmonicLoops(
        wiggle_dist=0.25,
    ),
)


c.add_action(
    wiggin_mito.actions.constraints.LoopBrushCylinderCompression(
        per_particle_volume=1.5*1.5*1.5,
    )
)


c.add_action(
    wiggin.actions.sim.LocalEnergyMinimization()
)


c.add_action(
    wiggin.actions.sim.BlockStep(
        # num_blocks=1000,
        block_size=10000
    ),
)


pprint.pprint(c.action_args)

c.auto_name_folder(root_data_folder='./data/helical_mitotic_chromosome/')

c.configure()

pprint.pprint(c.shared_config)
pprint.pprint(c.action_configs)

c.save_config()

c.run_init()

c.run_loop()
