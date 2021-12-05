from dataclasses import dataclass
import logging
from typing import Union, Tuple, Sequence, Any, Optional

import numpy as np

from .. import forces

from wiggin.core import SimAction

import looplib
import looplib.looptools
import looplib.random_loop_arrays


import polychrom
import polychrom.forces


logging.basicConfig(level=logging.INFO)


@dataclass
class GenerateSingleLayerLoops(SimAction):
    loop_size: float = 400
    loop_gamma_k: float = 1
    loop_spacing: int = 1
    chain_idxs: Optional[Sequence[int]] =  None

    _reads_shared = ['N', 'chains']
    _writes_shared = ['loops', 'backbone']

    def configure(self):
        out_shared = {}


        if self.chain_idxs is None:
            if "loop_n" in self.params:
                N = self.loop_n * self.loop_size
                out_shared["N"] = N
            else:
                N = self._shared["N"]
            chains = [(0, N, False)]

        else:
            if "chains" not in self._shared:
                raise ValueError("Chains are not configured!")
            if hasattr(self.chain_idxs, "__iter__"):
                chains = [
                    self._shared["chains"][i] for i in self.chain_idxs
                ]
            else:
                chains = [self._shared["chains"][int(self.chain_idxs)]]

        loops = []
        for start, end, is_ring in chains:
            chain_len = end - start
            if self.loop_gamma_k == 1:
                loops.append(
                    looplib.random_loop_arrays.exponential_loop_array(
                        chain_len,
                        self.loop_size,
                        self.loop_spacing
                    )
                )
            else:
                loops.append(
                    looplib.random_loop_arrays.gamma_loop_array(
                        chain_len,
                        self.loop_size,
                        self.loop_gamma_k,
                        self.loop_spacing,
                        min_loop_size=3
                    )
                )
            loops[0] += start
        loops = np.vstack(loops)

        out_shared["loops"] = (
            loops
            if "loops" not in out_shared
            else np.vstack([out_shared["loops"], loops])
        )

        try:
            out_shared["backbone"] = looplib.looptools.get_backbone(
                out_shared["loops"], N=N
            )
        except Exception:
            out_shared["backbone"] = None

        return out_shared


@dataclass
class GenerateTwoLayerLoops(SimAction):
    inner_loop_size: int = 400
    outer_loop_size: int = 400 * 4
    inner_loop_spacing: int = 1
    outer_loop_spacing: int = 1
    outer_inner_offset: int = 1
    inner_loop_gamma_k: float = 1
    outer_loop_gamma_k: float = 1
            
    _reads_shared = ['N']
    _writes_shared = ['loops', 'backbone']

        
    def configure(self):
        out_shared = {}

        if "outer_loop_n" in self.params:
            N = self.outer_loop_n * self.outer_loop_size
            out_shared["N"] = N
        elif "inner_loop_n" in self.params:
            N = self.inner_loop_n * self.inner_loop_size
            out_shared["N"] = N
        else:
            N = self._shared["N"]

        (
            outer_loops,
            inner_loops,
        ) = looplib.random_loop_arrays.two_layer_gamma_loop_array(
            N,
            self.outer_loop_size,
            self.outer_loop_gamma_k,
            self.outer_loop_spacing,
            self.inner_loop_size,
            self.inner_loop_gamma_k,
            self.inner_loop_spacing,
            self.outer_inner_offset,
        )
        loops = np.vstack([outer_loops, inner_loops])
        loops.sort()

        out_shared["loops"] = (
            loops
            if "loops" not in out_shared
            else np.vstack([out_shared["loops"], loops])
        )

        self.inner_loops = inner_loops
        self.outer_loops = outer_loops

        out_shared["backbone"] = looplib.looptools.get_backbone(
            outer_loops, N=N
        )

        return out_shared

