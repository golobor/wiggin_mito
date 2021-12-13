from dataclasses import dataclass
import logging
from typing import Union, Tuple, Sequence, Any, Optional # noqa: F401

import numpy as np

from .. import conformations

from wiggin.core import SimAction


logging.basicConfig(level=logging.INFO)


@dataclass
class HelicalLoopBrushConformation(SimAction):
    helix_radius: Optional[float] = None
    helix_turn_length: Optional[float] = None
    helix_step: Optional[float] = None
    axial_compression_factor: Optional[float] = None
    random_loop_orientations: bool = True
    
    _reads_shared = ['N', 'loops']
    _writes_shared = ['initial_conformation']

    def configure(self):
        out_shared = {}

        n_params = sum(
            [
                i is None
                for i in [
                    self.helix_radius,
                    self.helix_turn_length,
                    self.helix_step,
                    self.axial_compression_factor,
                ]
            ]
        )

        if n_params not in [0, 2]:
            raise ValueError(
                "Please specify 0 or 2 out of these four parameters: "
                "radius, turn_length, step and axis-to-backbone ratio"
            )

        if (self.helix_radius is not None) and (
            self.helix_step is not None
        ):
            helix_radius = self.helix_radius
            helix_step = self.helix_step
        elif (self.helix_turn_length is not None) and (
            self.helix_step is not None
        ):
            helix_step = self.helix_step
            helix_radius_squared = (
                (
                    (self.helix_turn_length) ** 2
                    - (self.helix_step) ** 2
                )
                / np.pi
                / np.pi
                / 2.0
                / 2.0
            )
            if helix_radius_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_radius = helix_radius_squared ** 0.5

        elif (self.helix_turn_length is not None) and (
            self.helix_radius is not None
        ):
            helix_radius = self.helix_radius
            helix_step_squared = (self.helix_turn_length) ** 2 - (
                2 * np.pi * helix_radius
            ) ** 2
            if helix_step_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_step = helix_step_squared ** 0.5

        elif (self.axial_compression_factor is not None) and (
            self.helix_radius is not None
        ):
            helix_radius = self.helix_radius
            helix_step = (
                2
                * np.pi
                * helix_radius
                / np.sqrt(self.axial_compression_factor ** 2 - 1)
            )

        elif (self.axial_compression_factor is not None) and (
            self.helix_turn_length is not None
        ):
            helix_step = (
                self.helix_turn_length
                / self.axial_compression_factor
            )
            helix_radius_squared = (
                ((self.helix_turn_length) ** 2 - (helix_step) ** 2)
                / np.pi
                / np.pi
                / 2.0
                / 2.0
            )
            if helix_radius_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_radius = helix_radius_squared ** 0.5
        elif (self.axial_compression_factor is not None) and (
            self.helix_step is not None
        ):
            helix_step = self.helix_step
            helix_turn_length = helix_step * self.axial_compression_factor
            helix_radius_squared = (
                ((helix_turn_length) ** 2 - (helix_step) ** 2)
                / np.pi
                / np.pi
                / 2.0
                / 2.0
            )
            if helix_radius_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_radius = helix_radius_squared ** 0.5
        else:
            helix_radius = 0
            helix_step = int(1e9)

        self.helix_step = helix_step
        self.helix_radius = helix_radius

        out_shared[
            "initial_conformation"
        ] = conformations.make_helical_loopbrush(
            L=self._shared["N"],
            helix_radius=helix_radius,
            helix_step=helix_step,
            loops=self._shared["loops"],
            random_loop_orientations=self.random_loop_orientations,
        )

        return out_shared


    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        sim.set_data(self._shared["initial_conformation"])

        return sim


@dataclass
class UniformHelicalLoopBrushConformation(SimAction):
    helix_radius: Optional[float] = None
    helix_step: Optional[float] = None
    axial_compression_factor: Optional[float] = None
    period_particles: Optional[float] = None
    loop_fold: str = "RW"
    chain_bond_length: float = 1.0
    
    _reads_shared = ['N', 'loops']
    _writes_shared = ['initial_conformation']        

    def configure(self):
        out_shared = {}

        n_params = sum(
            [
                i is not None
                for i in [
                    self.helix_radius,
                    self.helix_step,
                    self.axial_compression_factor,
                ]
            ]
        )

        if n_params not in [0, 2]:
            raise ValueError(
                "Please specify 0 or 2 out of these three parameters: "
                "radius, step and axis-to-backbone ratio"
            )

        if (self.helix_radius is not None) and (
            self.helix_step is not None
        ):
            helix_radius = self.helix_radius
            helix_step = self.helix_step
        elif (self.axial_compression_factor is not None) and (
            self.helix_radius is not None
        ):
            helix_radius = self.helix_radius
            helix_step = (
                2
                * np.pi
                * helix_radius
                / np.sqrt(self.axial_compression_factor ** 2 - 1)
            )

        elif (self.axial_compression_factor is not None) and (
            self.helix_step is not None
        ):
            helix_step = self.helix_step
            helix_turn_length = helix_step * self.axial_compression_factor
            helix_radius_squared = (
                (helix_turn_length ** 2 - helix_step ** 2) / np.pi / np.pi / 2.0 / 2.0
            )
            helix_radius = helix_radius_squared ** 0.5
        else:
            helix_radius = 0
            helix_step = int(1e9)

        self.helix_step = helix_step
        self.helix_radius = helix_radius

        out_shared[
            "initial_conformation"
        ] = conformations.make_uniform_helical_loopbrush(
            L=self._shared["N"],
            helix_radius=helix_radius,
            helix_step=helix_step,
            period_particles=self.period_particles,
            loops=self._shared["loops"],
            chain_bond_length=self.chain_bond_length,
            loop_fold=self.loop_fold,
        )

        return out_shared


    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        sim.set_data(self._shared["initial_conformation"])

        return sim

