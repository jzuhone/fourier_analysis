# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: .jupytext-sync-ipynb//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: py313
#     language: python
#     name: python3
# ---

# %%
from fourier_analysis import FourierAnalysis, window_data
import matplotlib.pyplot as plt
import yt

# %%
ds = yt.load_sample("GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0150")

# %%
slc = yt.SlicePlot(
    ds,
    "z",
    [("gas", "density"), ("gas", "kT")],
    width=(1.0, "Mpc"),
)
slc.annotate_velocity()
slc.show()

# %%
# This is the width of the grid on a side (here in kpc, but could be anything)
W = ds.arr([400.0] * 3, "kpc")
ddims = [128] * 3

# %%
c = ds.domain_center.to("kpc")

# %%
grid = ds.r[
    c[0] - W[0] / 2 : c[0] + W[0] / 2 : ddims[0] * 1j,
    c[1] - W[1] / 2 : c[1] + W[1] / 2 : ddims[1] * 1j,
    c[2] - W[2] / 2 : c[2] + W[2] / 2 : ddims[2] * 1j,
]

# %%
# This is a class I wrote to simplify stuff
fa = FourierAnalysis(W, ddims)

# %%
vx = grid[("gas", "velocity_x")].to_value("km/s")

# %%
# here I am applying a hanning window to the data to
# fix the fact that this is a non-periodic box and avoid aliasing
vxw = vx.copy()
window_data(vxw)

# %%
# Get the power spectrum of each spatial component
nbins = 60  # Number of bins for the power spectrum, it will
# use the min-max wavenumbers as boundaries
k, Pk = fa.make_powerspec(vx, nbins)
kw, Pkw = fa.make_powerspec(vxw, nbins)

# %%
# Now let's plot both the windowed and unwindowed power spectra
dx = W[0] / ddims[0]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(k, Pk, label="Unwindowed")
ax.loglog(kw, Pkw, label="Windowed")
#ax.loglog(k, 2e4*(k/1.0e-2)**(-11./3.), label="k^-11/3")
ax.set_xlabel("Wavenumber (k)")
ax.set_ylabel("Power Spectrum (Pk)")
ax.set_title("Power Spectrum")
ax.axvline(1.0/(2.0*dx), color="k", ls="--", label="Nyquist")
#ax.set_ylim(1.0e-2, 1.0e6)
ax.legend()
