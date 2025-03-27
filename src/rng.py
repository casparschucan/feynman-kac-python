import numpy as np
import os
import secrets


# Create a global dictionary to store thread-local RNGs
_process_local_rngs = {}


def get_rng():
    """Get a thread-local random number generator."""
    process_id = os.getpid()

    if process_id not in _process_local_rngs:
        # Create a new RNG for this thread
        # Use secrets for cryptographically strong randomness
        seed = secrets.randbits(64) ^ os.getpid()
        _process_local_rngs[process_id] = np.random.default_rng(seed)

    return _process_local_rngs[process_id]
