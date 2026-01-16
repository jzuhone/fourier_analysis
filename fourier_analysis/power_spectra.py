import numpy as np


def plaw_cutoffs(k, A, k0, k1, alpha):
    Pk = A * (1.0 + (k / k1) ** 2) ** (0.25 * alpha) * np.exp(-0.5 * (k / k0) ** 2)
    return Pk

