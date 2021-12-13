from dataclasses import dataclass
import logging
from typing import Sequence, Optional # noqa: F401

import numpy as np

from wiggin.core import SimAction

import looplib
import looplib.looptools
import looplib.random_loop_arrays


logging.basicConfig(level=logging.INFO)


@dataclass
class SingleLayerLoopPositions(SimAction):
    loop_size: float = 400
    loop_gamma_k: float = 1
    loop_spacing: int = 1
    chain_idxs: Optional[Sequence[int]] = None

    _reads_shared = ['N', 'chains']
    _writes_shared = ['loops', 'backbone']

    def configure(self):
        out_shared = {}

        if hasattr(self.chain_idxs, "__iter__"):
            chains = [
                self._shared["chains"][i] for i in self.chain_idxs
            ]
        elif self.chain_idxs is None:
            chains = self._shared['chains']
        else:
            chains = [self._shared["chains"][int(self.chain_idxs)]]

        loops = []
        for start, end, is_ring in chains:
            if end is None:
                end = self._shared['N']
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
                out_shared["loops"], N = self._shared['N']
            )
        except Exception:
            out_shared["backbone"] = None

        return out_shared


@dataclass
class TwoLayerLoopPositions(SimAction):
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

