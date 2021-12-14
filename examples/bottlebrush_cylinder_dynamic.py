import pprint

import wiggin
import wiggin_mito

c = wiggin.core.SimConstructor()

c.add_action(
    wiggin.actions.sim.InitializeSimulation(
        N=100000,
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
    wiggin_mito.actions.loops.SingleLayerLoopPositions(loop_size=400),
)

c.add_action(
    wiggin_mito.actions.conformations.HelicalLoopBrushConformation(
        helix_radius=1e-9, helix_step=1e9),
)

c.add_action(
    wiggin_mito.actions.constraints.DynamicLoopBrushCylinderCompression(
        ts_axial_compression=(200,300),
        ts_volume_compression=(100,200),
        axial_compression_factor=4,
    ),
)


c.add_action(
    wiggin_mito.actions.interactions.HarmonicLoops(
        wiggle_dist=0.25,
    ),
)


c.add_action(
    wiggin.actions.sim.LocalEnergyMinimization()
)

c.add_action(
    wiggin.actions.sim.BlockStep(
        num_blocks=1000,
    ),
)


pprint.pprint(c.action_args)


c.auto_name_folder(root_data_folder='./data/bottlebrush_cylinder_dynamic/')

c.configure()

pprint.pprint(c.shared_config)
pprint.pprint(c.action_configs)

c.save_config()

c.run_init()

c.run_loop()
