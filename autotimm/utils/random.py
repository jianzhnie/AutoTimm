"""Random wrapper."""
from __future__ import absolute_import
import random

import numpy as np
import torch


def setup_seed(seed=None):
    """Seed the generator for python builtin random, numpy.random,
    mxnet.random.

    This method is to control random state for mxnet related random functions.

    Note that this function cannot guarantee 100 percent reproducibility due to
    hardware settings.

    Parameters
    ----------
    a : int or 1-d array_like, optional
        Initialize internal state of the random number generator.
        If `seed` is not None or an int or a long, then hash(seed) is used instead.
        Note that the hash values for some types are nondeterministic.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
