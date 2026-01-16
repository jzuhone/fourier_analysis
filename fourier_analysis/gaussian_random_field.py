"""
3D Gaussian Random Field generation with a specified power spectrum.
"""

import numpy as np
from numba import njit, prange
from scipy.integrate import quad


sqrt2 = 2.0**0.5


def make_power_spec(user_func, **njit_kwargs):
    user_jit = njit(**njit_kwargs)(user_func)
    @njit
    def power_spec(k):
        return user_jit(k)  # user_func must be jittable!
    return power_spec


@njit(parallel=True, fastmath=True)
def compute_pspec(kx, ky, kz, nx, ny, nz, power_spec):
    sigma = np.zeros((nx, ny, nz))

    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                kk = np.sqrt(kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k])
                if kk == 0.0:
                    sigma[i, j, k] = 0.0
                else:
                    sigma[i, j, k] = power_spec(kk)

    return sigma


@njit(fastmath=True)
def enforce_hermitian_symmetry(f_hat):
    nx, ny, nz = f_hat.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz // 2 + 1):  # only loop over half
                i_, j_, k_ = (-i) % nx, (-j) % ny, (-k) % nz
                if (i, j, k) != (i_, j_, k_):
                    f_hat[i_, j_, k_] = np.conj(f_hat[i, j, k])
                else:
                    f_hat[i, j, k] = f_hat[i, j, k].real  # ensure real on Nyquist
    return f_hat


def make_gaussian_random_field(
    left_edge, right_edge, ddims, power_spec, f_rms, seed=None
):
    """
    Parameters
    ----------
    left_edge : array-like
        The lower edge of the box [kpc] for each of the dimensions.
    right_edge : array-like
        The upper edge of the box [kpc] for each of the dimensions.
    ddims : array-like
        The number of grids in each of the axes.
    l_min : float
        The minimum (dissipation) scale of the fluctuations.
    l_max : float
        The maximum (injection) scale of the fluctuations.
    alpha : float
        The power spectrum slope.
    f_rms : float
        The root mean square fluctuation amplitude.
    seed : int, optional
        Random seed for reproducibility.
    """

    prng = np.random.default_rng(seed=seed)

    left_edge = np.array(left_edge).astype("float64")
    right_edge = np.array(right_edge).astype("float64")
    ddims = np.array(ddims).astype("int")
    width = right_edge - left_edge
    deltas = width / ddims
    dx, dy, dz = deltas
    nx, ny, nz = ddims
    kx = np.arange(nx, dtype="float64")
    ky = np.arange(ny, dtype="float64")
    kz = np.arange(nz, dtype="float64")
    kx[kx > nx // 2] = kx[kx > nx // 2] - nx
    ky[ky > ny // 2] = ky[ky > ny // 2] - ny
    kz[kz > nz // 2] = kz[kz > nz // 2] - nz
    kx /= nx * dx
    ky /= ny * dy
    kz /= nz * dz

    pspec = make_power_spec(power_spec)

    def Ek(k):
        sigma = pspec(k)
        return 4.0 * np.pi * sigma * sigma * k * k

    Cn = f_rms**2 / quad(Ek, 0.0, 100.0, points=(0.1, 1.0, 10.0))[0] / np.prod(width)

    v_real = prng.normal(size=ddims)
    v_imag = prng.normal(size=ddims)
        
    sigma = compute_pspec(kx, ky, kz, nx, ny, nz, pspec) * np.sqrt(Cn)
    v_real *= sigma
    v_imag *= sigma

    v = v_real + 1j * v_imag
    v /= sqrt2

    v = enforce_hermitian_symmetry(v)

    v = np.fft.ifftn(v, norm="forward")

    return v.real
