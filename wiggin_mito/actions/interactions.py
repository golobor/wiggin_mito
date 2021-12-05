from dataclasses import dataclass
import logging
from typing import Union, Tuple, Sequence, Any, Optional

import numpy as np

from .. import forces

from wiggin.core import SimAction

import looplib
import looplib.looptools

import polychrom
import polychrom.forces


logging.basicConfig(level=logging.INFO)



@dataclass
class AddLoops(SimAction):
    wiggle_dist: float = 0.05
    bond_length: float = 1.0

    _shared = dict(loops=None)


    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        sim.add_force(
            polychrom.forces.harmonic_bonds(
                sim_object=sim,
                bonds=self._shared["loops"],
                bondWiggleDistance=self.wiggle_dist,
                bondLength=self.bond_length,
                name="loop_harmonic_bonds",
                override_checks=True,
            )
        )


@dataclass
class AddRootLoopSeparator(SimAction):
    wiggle_dist: float = 0.1

    _shared = dict(loops=None)


    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        loops = self._shared["loops"]
        root_loops = loops[looplib.looptools.get_roots(loops)]
        root_loop_spacers = np.vstack([root_loops[:-1][:, 1], root_loops[1:][:, 0]]).T
        root_loop_spacer_lens = root_loop_spacers[:, 1] - root_loop_spacers[:, 0]

        sim.add_force(
            forces.harmonic_bonds(
                sim_object=sim,
                bonds=root_loop_spacers,
                bondWiggleDistance=self.wiggle_dist,
                bondLength=root_loop_spacer_lens,
                name="RootLoopSpacers",
                override_checks=True,
            )
        )


@dataclass
class AddBackboneStiffness(SimAction):
    k: float = 1.5

    _shared = dict(backbone=None)
        

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        bb_list = sorted(self._shared["backbone"])
        triplets = [bb_list[i : i + 3] for i in range(len(bb_list) - 2)]
        sim.add_force(
            forces.angle_force(
                sim_object=sim,
                triplets=triplets,
                k=self.k,
                theta_0=np.pi,
                name="backbone_stiffness",
                override_checks=True,
            )
        )


