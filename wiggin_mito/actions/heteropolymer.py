from dataclasses import dataclass
import logging
import numbers
from typing import Union, Tuple, Sequence, Any, Optional

import numpy as np

from .. import forces

from wiggin.core import SimAction

import polychrom
import polychrom.forces
import polychrom.forcekits


logging.basicConfig(level=logging.INFO)


@dataclass
class GenerateRandomBlockParticleTypes(SimAction):
    avg_block_lens: Sequence[int] = (2, 2)
    
    _shared = dict(N=None)


    def configure(self):
        out_shared = {}

        # This solution is slow-ish (1 sec for 1e6 particles), but simple
        N = self._shared['N']
        avg_block_lens = self.avg_block_lens
        n_types = len(avg_block_lens)
        particle_types = np.full(N, -1)

        p, new_p, t = 0, 0, 0
        while new_p <= N:
            new_p = p + np.random.geometric(1 / avg_block_lens[t])
            particle_types[p : min(new_p, N)] = t
            t = (t + 1) % n_types
            p = new_p

        out_shared["particle_types"] = particle_types

        return out_shared



@dataclass
class AddChainsSelectiveRepAttr(SimAction):
    chains: Any = ((0, None, 0),)
    bond_length: float =1.0
    wiggle_dist: float =0.025
    stiffness_k: Optional[float] = None
    repulsion_e: Optional[float] = 2.5,  # TODO: implement np.in
    attraction_e: Optional[float] = None
    attraction_r: Optional[float] = None
    selective_attraction_e: Optional[float] = None
    particle_types: Any = None
    except_bonds: bool = False
    
    _shared = dict()
        

    def configure(self):
        out_shared = {}

        if hasattr(self.chains, "__iter__") and hasattr(
            self.chains[0], "__iter__"
        ):
            out_shared["chains"] = self.chains
        elif hasattr(self.chains, "__iter__") and isinstance(
            self.chains[0], numbers.Number
        ):
            edges = np.r_[0, np.cumsum(self.chains)]
            chains = [(st, end, False) for st, end in zip(edges[:-1], edges[1:])]
            self.chains = chains
            out_shared["chains"] = chains

        return out_shared

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        nonbonded_force_func = forces.homotypic_quartic_repulsive_attractive
        nonbonded_force_kwargs = dict(
            repulsionEnergy=self.repulsion_e,
            repulsionRadius=1.0,
            attractionEnergy=self.attraction_e,
            attractionRadius=self.attraction_r,
            particleTypes=self.particle_types,
            selectiveAttractionEnergy=self.selective_attraction_e,
        )

        sim.add_force(
            polychrom.forcekits.polymer_chains(
                sim,
                chains=self._shared["chains"],
                bond_force_func=polychrom.forces.harmonic_bonds,
                bond_force_kwargs={
                    "bondLength": self.bond_length,
                    "bondWiggleDistance": self.wiggle_dist,
                },
                angle_force_func=(
                    None if self.stiffness_k is None else polychrom.forces.angle_force
                ),
                angle_force_kwargs={"k": self.stiffness_k},
                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,
                except_bonds=self.except_bonds,
            )
        )


@dataclass
class AddChainsHeteropolymerRepAttr(SimAction):
    chains: Any = ((0, None, 0),)
    bond_length: float =1.0
    wiggle_dist: float =0.025
    stiffness_k: Optional[float] = None
    repulsion_e: Optional[float] = 2.5,  # TODO: implement np.in
    attraction_e: Optional[float] = None
    attraction_r: Optional[float] = None
    particle_types: Any = None
    except_bonds: bool = False
    
    _shared = dict()        

    def configure(self):
        out_shared = {}

        if hasattr(self.chains, "__iter__") and hasattr(
            self.chains[0], "__iter__"
        ):
            out_shared["chains"] = self.chains
        elif hasattr(self.chains, "__iter__") and isinstance(
            self.chains[0], numbers.Number
        ):
            edges = np.r_[0, np.cumsum(self.chains)]
            chains = np.array(
                [(st, end, False) for st, end in zip(edges[:-1], edges[1:])],
                dtype=np.object,
            )
            self.chains = chains
            out_shared["chains"] = chains

        return out_shared

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from self.selfame] and self._shared

        nonbonded_force_func = forces.heteropolymer_quartic_repulsive_attractive
        nonbonded_force_kwargs = dict(
            repulsionEnergy=self.repulsion_e,
            repulsionRadius=1.0,
            attractionEnergies=self.attraction_e,
            attractionRadius=self.attraction_r,
            particleTypes=(
                self._shared["particle_types"]
                if self.particle_types is None
                else self.particle_types
            ),
        )

        sim.add_force(
            polychrom.forcekits.polymer_chains(
                sim,
                chains=self._shared["chains"],
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    "bondLength": self.bond_length,
                    "bondWiggleDistance": self.wiggle_dist,
                },
                angle_force_func=(
                    None if self.stiffness_k is None else polychrom.forces.angle_force
                ),
                angle_force_kwargs={"k": self.stiffness_k},
                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,
                except_bonds=self.except_bonds,
            )
        )


