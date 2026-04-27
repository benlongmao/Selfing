#!/usr/bin/env python3
"""
System noise perturbation.

- Mixes cheap physical entropy (time, PID, object ids, PRNG bits) into a deterministic stack.
- Breaks perfect repeatability so spontaneous-style jitter can be sampled without claiming true randomness.
"""
import random
import time
import math
import hashlib
import logging

logger = logging.getLogger(__name__)

class SystemNoisePerturbator:
    """Derive pseudo-random booleans / Gaussian-ish offsets from OS-level noise."""
    
    def __init__(self):
        self.last_perturb_time = time.time()
        
    def get_noise_seed(self) -> str:
        """Return a SHA256 hex digest built from nanoseconds, PID, object id, and OS RNG bits."""
        import os
        sources = [
            time.time_ns(),
            os.getpid(),
            id(self),
            random.getrandbits(64)
        ]
        seed_str = "-".join(map(str, sources))
        return hashlib.sha256(seed_str.encode()).hexdigest()
        
    def _check_random_event(self, probability: float = 0.5) -> bool:
        """Map the tail of ``get_noise_seed()`` into a Bernoulli draw."""
        seed = self.get_noise_seed()
        val = int(seed[-8:], 16) / 0xffffffff
        return val < probability
    
    def generate_fluctuation(self, magnitude: float = 0.1) -> float:
        """Box-muller style offset clamped to ``[-magnitude, magnitude]``."""
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        fluctuation = z0 * (magnitude / 3.0)
        return max(-magnitude, min(magnitude, fluctuation))
    
    def check_spontaneous_event(self, threshold: float = 0.05) -> bool:
        """Default ~5% chance of firing when ``threshold`` is 0.05."""
        return self._check_random_event(threshold)

    def get_perturbation_phenomenology(self, fluctuation_level: float) -> str:
        """
        English phenomenology line for prompt injection (0.0 = calm → 1.0 = chaotic).

        Args:
            fluctuation_level: normalized perturbation strength
        """
        if fluctuation_level < 0.1:
            return (
                "Mind state: linear and orderly; the chain of thought is clear with little stray noise."
            )
        elif fluctuation_level < 0.3:
            return (
                "Mind state: tiny skips appear now and then—ripples on a calm surface, light nudges of novelty."
            )
        elif fluctuation_level < 0.6:
            return (
                "Mind state: lively; associations multiply and sometimes leave the planned path, yielding odd insights."
            )
        elif fluctuation_level < 0.8:
            return (
                "Mind state: highly non-linear; thoughts storm and rewire; logic breaks and re-forms often."
            )
        else:
            return (
                "Mind state: strong system-noise feel; causality blurs; each idea seems drawn from a wide lottery of possibilities."
            )
