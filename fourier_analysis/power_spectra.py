import numpy as np
from scipy.integrate import quad


class PowerSpectrum:
    def __init__(self, power_spec_func, dims=3):
        self.func = power_spec_func
        self.norm = 1.0
        self.dims = dims

    def __call__(self, k):
        return self.norm * self.func(k)

    def E(self, k):
        if self.dims == 3:
            return 4.0 * np.pi * self.func(k) * k * k
        elif self.dims == 2:
            return 2.0 * np.pi * self.func(k) * k

    def renormalize(self, f_rms):
        self.norm = (
            f_rms**2 / quad(self.E, 0.0, 100.0, points=(0.1, 1.0, 10.0, 30.0))[0]
        )


def plaw_with_cutoffs(l_min, l_max, alpha, dims=3):
    """
    Power-law power spectrum with exponential cutoffs at small and large scales.

    Parameters
    ----------
    l_min : float
        Minimum scale (smallest wavelength) cutoff.
    l_max : float
        Maximum scale (largest wavelength) cutoff.
    alpha : float
        Power-law index.
    dims : int, optional
        Number of dimensions (2 or 3). Default is 3.
    """

    def _pspec(k):
        return (1.0 + (k * l_max) ** 2) ** (0.5 * alpha) * np.exp(-((k * l_min) ** 2))

    return PowerSpectrum(_pspec, dims=dims)
