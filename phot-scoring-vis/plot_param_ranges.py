"""
Generate a four panel plot of the acceptable parameter ranges for
photometry scoring in the luminosity vs. time parameter space
"""
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from otter import Otter
from astropy import units as u
#from scoring.vet_phot import FILTER_EFF_FREQ
# from django.conf import settings
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

FILTER_EFF_FREQ = {
    'u': 8.468e14 * u.Hz,
    'g': 6.289e14 * u.Hz,
    'r': 4.832e14 * u.Hz,
    'i': 3.948e14 * u.Hz,
    'z': 3.343e14 * u.Hz,
    'y': 3.043e14 * u.Hz,
    'U': 8.468e14 * u.Hz,
    'B': 6.810e14 * u.Hz,
    'V': 5.483e14 * u.Hz,
    'R': 4.610e14 * u.Hz,
    'I': 3.807e14 * u.Hz,
    'c': 5.4e14 * u.Hz, # ATLAS cyan
    'o': 4.8e14 * u.Hz, # ATLAS orange
    'G': 5.5e14 * u.Hz, # Gaia G-band
    'w': 5.208e14 * u.Hz, # Pan-STARRS w (wide)
    'F070W': 4.310e14 * u.Hz, # JWST
    'F090W': 3.362e14 * u.Hz,
    'F115W': 2.629e14 * u.Hz,
    'F150W': 2.019e14 * u.Hz,
    'F182M': 1.631e14 * u.Hz,
    'F200W': 1.526e14 * u.Hz,
    'F250M': 1.199e14 * u.Hz,
    'F277W': 1.101e14 * u.Hz,
    'F300M': 1.006e14 * u.Hz,
    'F335M': 0.894e14 * u.Hz,
    'F356W': 0.850e14 * u.Hz,
    'F360M': 0.829e14 * u.Hz,
    'F444W': 0.690e14 * u.Hz,
    'F560W': 0.537e14 * u.Hz,
    'F770W': 0.399e14 * u.Hz,
    'F1000W': 0.303e14 * u.Hz,
    'F1130W': 0.265e14 * u.Hz,
    'F1280W': 0.236e14 * u.Hz,
    'F1500W': 0.201e14 * u.Hz,
    'F1800W': 0.168e14 * u.Hz,
    'F2100W': 0.146e14 * u.Hz,
    'F2550W': 0.119e14 * u.Hz,
    'F062': 4.962e14 * u.Hz, # Roman
    'F087': 3.494e14 * u.Hz,
    'F106': 2.876e14 * u.Hz,
    'F129': 2.355e14 * u.Hz,
    'F146': 2.355e14 * u.Hz,
    'F158': 1.929e14 * u.Hz,
    'F213': 1.421e14 * u.Hz,
}

cosmo = FlatLambdaCDM(H0=69.6 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) #settings.COSMO

db = Otter()

RNG = np.random.default_rng()

colors = ["#0072B2", "#E69F00", "#009E73"]

# slightly modified versions of the parameter ranges in MODELS_BY_NAME
# since we don't want a bunch of negative infinities
@dataclass
class Params:
    lum_max: list
    peak_time: list
    decay_rate: list = None
    
def _magrate_to_dexrate(magrate):
    """mag/day (positive = fading) -> dex/day decline in log10(L)"""
    return 0.4 * magrate

MODELS_BY_NAME = {
    "KN": Params(
        lum_max=[1e40, 1e43],  # log10(erg/s)
        peak_time=[0, 4],
        decay_rate=[0.1, 10],  # dex/day
    ),
    "KN-in-SN": Params(
        lum_max=[5e41, 1e44],
        peak_time = [0,35],
        decay_rate = [-2, 0.1]
    ),
    "Super-KN": Params(
        lum_max=[1e41, 1e43],
        peak_time = [10, 70],
        # decay_rate = [-np.inf, np.inf]
    )
}
################################################
## FUNCTIONS ###################################
################################################ 
def make_corner_axes(fig, n, space=0.05):
    """Create the (n-1) x (n-1) grid of axes for an n-parameter corner plot
    (no 1D diagonal panels) on a Figure or SubFigure."""
    return fig.subplots(n - 1, n - 1, gridspec_kw=dict(wspace=space, hspace=space),
                        squeeze=False)
 
 
def corner_ranges(axes, regions, param_labels, region_labels=None, colors=colors,
                  bounds=None, alpha=0.35, log_params=(), symlog_params=(),
                  linthresh=0.01, missing="unconstrained"):
    """
    Draw rectangular "allowed" regions on an existing n x n grid of axes.
 
    axes         : n x n array of axes (e.g. from make_corner_axes)
    regions      : list of regions; each is a list with one entry per parameter,
                   a (min, max) tuple or None if the region lacks that parameter
    param_labels : names of the parameters
    region_labels: legend names, one per region
    colors       : colours, one per region
    bounds       : list of (min, max) plot limits per parameter;
                   default = min/max over all regions
    log_params   : parameter indices to show on a log axis
    symlog_params: parameter indices to show on a symlog axis
    missing      : "unconstrained" (span the full axis, hatched) or "skip"
    """
    n = len(param_labels)
    colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    region_labels = region_labels or [f"Region {k+1}" for k in range(len(regions))]
 
    for k, r in enumerate(regions):
        if len(r) != n:
            raise ValueError(f"{region_labels[k]} has {len(r)} ranges but there are "
                             f"{n} parameters; use None for missing ones.")
 
    if bounds is None:
        bounds = []
        for p in range(n):
            given = [r[p] for r in regions if r[p] is not None]
            if not given:
                raise ValueError(f"No region constrains '{param_labels[p]}'; "
                                 "pass bounds explicitly.")
            bounds.append((min(g[0] for g in given), max(g[1] for g in given)))
 
    def set_scale(ax, p, which):
        setter = ax.set_xscale if which == "x" else ax.set_yscale
        if p in log_params:
            setter("log")
        elif p in symlog_params:
            setter("symlog", linthresh=linthresh)
 
    first_patch, labelled = {}, set()
    for i in range(1, n):           # y parameter (row i-1)
        for j in range(n - 1):      # x parameter (column j)
            ax = axes[i - 1, j]
            if j >= i:
                ax.axis("off")
                continue
 
            for k, region in enumerate(regions):
                c = colors[k % len(colors)]
                xr, yr = region[j], region[i]
                incomplete = xr is None or yr is None
                if incomplete and missing == "skip":
                    continue
                xr = xr if xr is not None else bounds[j]
                yr = yr if yr is not None else bounds[i]
                (x0, x1), (y0, y1) = xr, yr
                patch = ax.add_patch(Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, facecolor=c, edgecolor=c,
                    alpha=alpha, lw=1.5, hatch="//" if incomplete else None))
                first_patch.setdefault(k, patch)
                if not incomplete and k not in labelled:   # legend: prefer unhatched
                    patch.set_label(region_labels[k])
                    labelled.add(k)
 
            set_scale(ax, j, "x")
            ax.set_xlim(bounds[j])
            set_scale(ax, i, "y")
            ax.set_ylim(bounds[i])
 
            if i == n - 1:
                ax.set_xlabel(param_labels[j])
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(param_labels[i])
            else:
                ax.tick_params(labelleft=False)
 
    for k, patch in first_patch.items():    # regions only ever drawn hatched
        if k not in labelled:
            patch.set_label(region_labels[k])
 
    return axes
 
 
def add_point(axes, point, color="k", marker="*", size=120, label=None, **kwargs):
    """Mark one point (one value per parameter) on every pairwise panel."""
    n = len(point)
    labelled = False
    for i in range(1, n):
        for j in range(i):
            axes[i - 1, j].scatter(point[j], point[i], color=color, marker=marker,
                                   s=size, zorder=5,
                                   label=None if labelled else label, **kwargs)
            labelled = True
 
 
def corner_legend(axes, ax=None, loc="upper right", frameon=False, **kwargs):
    """
    One legend for everything labelled in the corner plot (regions and points).
    By default it goes in the empty top-right panel; pass ax= to put it elsewhere.
    """
    entries = {}
    for a in np.ravel(axes):
        for h, l in zip(*a.get_legend_handles_labels()):
            if l and not l.startswith("_") and l not in entries:
                entries[l] = h
    target = ax if ax is not None else axes[0, -1]
    handles, labels = list(entries.values()), list(entries.keys())
    handles += [Line2D([0], [0], linestyle=":", color="k")]
    labels += ["Broken Powerlaw"]
    return target.legend(handles, labels,
                         loc=loc, frameon=frameon, **kwargs)
 

    
def line_density(curves, t_edges, logL_edges, step=0.25):
    """
    Count how many curves pass through each (t, log10 L) cell.
 
    curves     : list of (t, L) array pairs, one per black line
    t_edges    : bin edges in days
    logL_edges : bin edges in log10(L)
    step       : sampling step along each line, in units of one cell.
                 Must be < 1 so steep segments don't skip cells.
    """
    nx, ny = len(t_edges) - 1, len(logL_edges) - 1
    counts = np.zeros((nx, ny), dtype=int)
    dx, dy = np.diff(t_edges).mean(), np.diff(logL_edges).mean()
 
    for t, L in curves:
        t, L = np.asarray(t, float), np.asarray(L, float)
        ok = np.isfinite(t) & np.isfinite(L) & (L > 0)
        t, y = t[ok], np.log10(L[ok])
        if len(t) < 2:
            continue
 
        # Work in "cell units" so the sampling is dense in BOTH directions.
        # The lines are straight on a log-y plot, so interpolate in log10(L).
        u = (t - t_edges[0]) / dx
        v = (y - logL_edges[0]) / dy
        seg = np.hypot(np.diff(u), np.diff(v))
        s = np.concatenate([[0], np.cumsum(seg)])
        s_new = np.arange(0, s[-1] + step, step)
        u_s = np.interp(s_new, s, u)
        v_s = np.interp(s_new, s, v)
 
        ix, iy = np.floor(u_s).astype(int), np.floor(v_s).astype(int)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        # Each line counts at most once per cell
        cells = np.unique(np.stack([ix[inside], iy[inside]]), axis=1)
        counts[cells[0], cells[1]] += 1
 
    return counts

def lc_model(t, rise_slope, decay_dex, logL_peak, t_peak):
    """Power-law rise to peak, then exponential (linear-in-mag) decline. LOG SPACE."""
    with np.errstate(divide="ignore"):
        rise = logL_peak + rise_slope * np.log10(t / t_peak)
    decay = logL_peak - decay_dex * (t - t_peak)
    return np.where(t < t_peak, rise, decay)

def valid_y(t, model_name="KN", num=1_000):
    params = MODELS_BY_NAME[model_name]
    
    # select randomly from the params
    peak_lum = RNG.uniform(*params.lum_max, size=num)
    peak_time = RNG.uniform(*params.peak_time, size=num)
    decay_rate = RNG.uniform(*params.decay_rate, size=num)
    rise_rate = peak_lum/peak_time
    
    # calculate some nominal broken powerlaw from these values
    x = t[:, None]
    y = lc_model(x, rise_rate, decay_rate, peak_lum, peak_time)
    
    return y #np.min(y,axis=1), np.max(y,axis=1)


def plot_lc(t, ys, ax=None, ytop=1e44, ybottom=1e40):
    if ax is None:
        fig, ax = plt.subplots()

    upper = 10**np.max(ys)
    ax.plot(t, upper, color="C0", lw=2)
    ax.fill_between(t, upper, ytop, facecolor="none", edgecolor="0.6",
                   hatch="///", lw=0)
    for y in 10**ys.T:
       ax.plot(t, y, alpha=0.1, lw=1, color="k")


    ax.set_ylim(ybottom, ytop)
    ax.set_xlim(min(t), max(t))

    # label positions: middle of the time range, geometric midpoint of each region
    i = len(t) // 4
    text_kw = dict(color="red", fontweight="bold", ha="center", va="center")
    ax.text(t[i], np.sqrt(upper[i] * ytop), "EXCLUDED",
           bbox=dict(facecolor="white", edgecolor="none", pad=2), **text_kw)
    ax.text(t[i], 1e42, "ALLOWED", **text_kw)
    return ax

def add_from_otter(ax, name="1998bw", filtname="R", label=None, color="#FF474C", overwrite=False):

    # metadata
    bw = db.get_meta(names=name)[0]
    z = bw.get_redshift()
    disc_date = bw.get_discovery_date().mjd

    fname = f"{name}_otter.csv"
    if os.path.exists(fname) and not overwrite:
        df = pd.read_csv(fname) # cache the data
    else:
        df = db.get_phot(names=name, return_type="pandas", flux_unit="mag(AB)")
        df.to_csv(fname)
        
    rlc = df[df.filter_name == filtname].sort_values("converted_date")

    t = rlc.converted_date - disc_date
    rlc["phase"] = t
    # f = (rlc.converted_flux.values*u.Jy * FILTER_EFF_FREQ[filtname]).to(u.erg/u.s/u.cm**2)
    # lum = (f * 4*np.pi*cosmo.luminosity_distance(z)**2).to(u.erg/u.s).value

    ax.errorbar(t, rlc.converted_flux, yerr=rlc.converted_flux_err, marker="*", linestyle="none", color=color, label=label, markersize=12)

    return rlc
    
def add_98bw(ax, **kwargs):
    return add_from_otter(ax, **kwargs)

def add_04et(ax, **kwargs):
    return add_from_otter(ax, name="2004et", label="SN2004et\n"+r"(R-band $\nu L_\nu$)", color=colors[-1], **kwargs)

def add_17gfo(ax):

    df = pd.read_csv("AT2017gfo-lightcurve.csv")
    dist = 39.5*u.Mpc
    rlc = df[df.Band == "r"]
    t = rlc.Phase
    # f = ((rlc.mag.values*u.ABmag).to(u.Jy) * FILTER_EFF_FREQ["r"]).to(u.erg/u.s/u.cm**2)
    # lum = (f * 4*np.pi*dist**2).to(u.erg/u.s).value
    # lum_err = lum * rlc.e_mag/rlc.mag
    
    ax.errorbar(
        t, rlc.mag,
        marker="o",
        linestyle="none",
        yerr=rlc.e_mag,
        # label="AT2017gfo\n"+r"(r-Band)",
        color="k",
        markersize=9
    )

    return rlc
    
################################################
## The code below this line will execute upon ##
## a `run plot_param_ranges.py` in the django ##
## shell for TROVE #############################
################################################

# fig,ax = plt.subplots()

# # KN
# t = np.linspace(0, 25)
# y = valid_y(t)

# ax = plot_lc(t, y, ax)

# ax.set_ylabel("Optical Luminosity [erg/s]")
# ax.set_xlabel("Days Since GW Trigger")
# ax.set_yscale("log")
# add_17gfo(ax)
# add_98bw(ax)
# #add_04et(ax)
# ax.legend(loc="lower right")
# fig.show()
