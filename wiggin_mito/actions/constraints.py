from dataclasses import dataclass
import logging
from typing import Union, Tuple, Sequence, Any, Optional # noqa: F401

import numpy as np

import wiggin_mito.forces

import wiggin
from wiggin.core import SimAction

import looplib
import looplib.looptools

import polychrom
import polychrom.forces

logging.basicConfig(level=logging.INFO)


@dataclass
class BackboneTethering(SimAction):
    k: float=15  

    _reads_shared = ['backbone']

    def run_init(self, sim):

        sim.add_force(
            polychrom.forces.tether_particles(
                sim_object=sim,
                particles=self._shared["backbone"],
                k=self.k,
                positions="current",
                name="wm_tether_backbone",
            )
        )


@dataclass
class BackboneAngularTethering(SimAction):
    angle_wiggle: float = np.pi / 16

    _reads_shared = ['backbone']

    def run_init(self, sim):

        sim.add_force(
            wiggin_mito.forces.angular_tether_particles(
                sim_object=sim,
                particles=self._shared["backbone"],
                angle_wiggle=self.angle_wiggle,
                angles="current",
                name="wm_tether_backbone_angle",
            )
        )



@dataclass
class RootLoopBaseAngularTethering(SimAction):
    angle_wiggle: float = np.pi / 16

    _reads_shared = ['loops']

    def run_init(self, sim):

        loops = self._shared["loops"]
        root_loops = loops[looplib.looptools.get_roots(loops)]
        root_loop_particles = sorted(np.unique(root_loops))

        sim.add_force(
            wiggin_mito.forces.angular_tether_particles(
                sim_object=sim,
                particles=root_loop_particles,
                angle_wiggle=self.angle_wiggle,
                angles="current",
                name="wm_tether_root_loops_angle",
            )
        )


@dataclass
class TetherTips(SimAction):
    k: Union[float, Tuple[float, float, float]] = (0, 0, 5)
    particles: Sequence[int] = (0, -1)
    positions: Any = "current"
            

    def run_init(self, sim):

        sim.add_force(
            polychrom.forces.tether_particles(
                sim_object=sim,
                particles=self.particles,
                k=self.k,
                positions=self.positions,
                name='wm_tether_tips'
            )
        )


@dataclass
class LoopBrushCylinderCompression(SimAction):
    k: Optional[float] = 1.0
    z_min: Optional[Union[float, str]] = None
    z_max: Optional[Union[float, str]] = None
    r: Optional[float] = None 
    per_particle_volume: Optional[float] = 1.25 * 1.25 * 1.25
        
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
        # elif (self.r is None) and (
        #     self.per_particle_volume is not None
        # ):
        #     self.r = np.sqrt(
        #         self._shared["N"]
        #         * self.per_particle_volume
        #         / (self.z_max - self.z_min)
        #         / np.pi
        #     )

        return {}

    def run_init(self, sim):

        sim.add_force(
            wiggin_mito.forces.cylindrical_confinement(
                sim_object=sim,
                r=self.r,
                per_particle_volume=self.per_particle_volume,
                bottom=self.z_min,
                top=self.z_max,
                k=self.k,
                name='wm_cylindrical_confinement'
            )
        )


@dataclass
class DynamicLoopBrushCylinderCompression(SimAction):
    ts_axial_compression: Optional[Tuple[int, int]] = (100, 200)
    ts_volume_compression: Optional[Tuple[int, int]] = (100, 200)
    per_particle_volume: Optional[float] = 1.25 * 1.25 * 1.25
    k_confinement: Optional[float] = 1.0
    axial_length_final: Optional[float] = None

    _reads_shared = ['N', 'initial_conformation']

    def spawn_actions(self):
        new_actions = []
        N = self._shared["N"]
        coords = self._shared["initial_conformation"]
        bottom_init = coords[:, 2].min()
        top_init = coords[:, 2].max()
        r_init = ((coords[:, :2] ** 2).sum(axis=1) ** 0.5).max()
        ppv_init = np.pi * r_init * r_init * (top_init - bottom_init) / N
        axial_length_final = self.axial_length_final

        new_actions.append(
            LoopBrushCylinderCompression(
                k=self.k_confinement,
                z_min=bottom_init,
                z_max=top_init,
                per_particle_volume=ppv_init
            )
        )

        new_actions.append(TetherTips())
      
        new_actions.append(
            wiggin.actions.sim.UpdateGlobalParameter(
                force='wm_cylindrical_confinement',
                param='top',
                ts=self.ts_axial_compression,
                vals=[top_init, bottom_init + axial_length_final],
                #vals=[bottom_init, (bottom_init + top_init) / 2 + axial_length_final / 2]
            ).rename('UpdateConfinementTop')
        )
        
        new_actions.append(
            wiggin.actions.sim.UpdatePerParticleParameter(
                force='wm_tether_tips',
                parameter_name='z0',
                term_index=1,
                ts=self.ts_axial_compression,
                vals=[top_init, bottom_init+axial_length_final],
            ).rename('UpdateTopTipTethering')
        )

        new_actions.append(
            wiggin.actions.sim.UpdateGlobalParameter(
                force='wm_cylindrical_confinement',
                param='ppv',
                ts=self.ts_volume_compression,
                vals=[ppv_init, self.per_particle_volume],
                power=1/4,
            ).rename('UpdateConfinementVolume')
        )

        return new_actions
