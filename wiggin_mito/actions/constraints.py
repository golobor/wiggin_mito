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
class AddBackboneTethering(SimAction):
    k: float=15  

    _reads_shared = ['backbone']

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        sim.add_force(
            polychrom.forces.tether_particles(
                sim_object=sim,
                particles=self._shared["backbone"],
                k=self.k,
                positions="current",
                name="tether_backbone",
            )
        )


@dataclass
class AddBackboneAngularTethering(SimAction):
    angle_wiggle: float = np.pi / 16

    _reads_shared = ['backbone']

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        sim.add_force(
            forces.angular_tether_particles(
                sim_object=sim,
                particles=self._shared["backbone"],
                angle_wiggle=self.angle_wiggle,
                angles="current",
                name="tether_backbone_angle",
            )
        )



@dataclass
class AddRootLoopAngularTethering(SimAction):
    angle_wiggle: float = np.pi / 16

    _reads_shared = ['loops']

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        loops = self._shared["loops"]
        root_loops = loops[looplib.looptools.get_roots(loops)]
        root_loop_particles = sorted(np.unique(root_loops))

        sim.add_force(
            forces.angular_tether_particles(
                sim_object=sim,
                particles=root_loop_particles,
                angle_wiggle=self.angle_wiggle,
                angles="current",
                name="tether_root_loops_angle",
            )
        )


@dataclass
class AddTipsTethering(SimAction):
    k: Union[float, Tuple[float, float, float]] = (0, 0, 5)
    particles: Sequence[int] = (0, -1)
    positions: Any = "current"
            

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        sim.add_force(
            forces.tether_particles(
                sim_object=sim,
                particles=self.particles,
                k=self.k,
                positions=self.positions,
            )
        )


@dataclass
class AddStaticCylinderCompression(SimAction):
    k: Optional[float] = 1.0
    z_min: Optional[Union[float, str]] = None
    z_max: Optional[Union[float, str]] = None
    r: Optional[float] = None 
    per_particle_volume: Optional[float] =1.5 * 1.5 * 1.0
        
    _reads_shared = ['N', 'backbone', 'initial_conformation']

    def configure(self):

        if (self.z_min is None) != (self.z_max is None):
            raise ValueError(
                "Both z_min and z_max have to be either specified or left as None."
            )

        coords = self._shared["initial_conformation"]
        
        if self.z_min is None:
            self.z_min = coords[:, 2].min()
        elif self.z_min == "bb":
            self.z_min = coords[self._shared["backbone"]][:, 2].min()
        else:
            self.z_min = self.z_min

        if self.z_max is None:
            self.z_max = coords[:, 2].max()
        elif self.z_max == "bb":
            self.z_max = coords[self._shared["backbone"]][:, 2].max()
        else:
            self.z_max = self.z_max

        if (self.r is not None) and (
            self.per_particle_volume is not None
        ):
            raise ValueError("Please specify either r or per_particle_volume.")
        elif (self.r is None) and (
            self.per_particle_volume is None
        ):
            coords = self._shared["initial_conformation"]
            self.r = ((coords[:, :2] ** 2).sum(axis=1) ** 0.5).max()
        elif (self.r is None) and (
            self.per_particle_volume is not None
        ):
            self.r = np.sqrt(
                self._shared["N"]
                * self.per_particle_volume
                / (self.z_max - self.z_min)
                / np.pi
            )

        return {}

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        sim.add_force(
            forces.cylindrical_confinement_2(
                sim_object=sim,
                r=self.r,
                top=self.z_max,
                bottom=self.z_min,
                k=self.k,
            )
        )

