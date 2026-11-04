"""
Generate mock Rubin-LSST lightcurve for the two-component KN model in
Villar+17, at some coordinates and with some host galaxy within the
localization volume of GW241109_115924 (S241109bn).

Use lightcurvelynx and redback to generate simulations.

Based largely on the following example:
https://lightcurvelynx.readthedocs.io/en/latest/notebooks/pre_executed/redback_example.html
"""

import numpy as np
import pandas as pd


# redback imports
from redback import model_library

# lightcurvelynx imports
from lightcurvelynx import _LIGHTCURVELYNX_BASE_DATA_DIR
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.survey_info import SurveyInfo
from lightcurvelynx.models.redback_models import RedbackWrapperModel
from lightcurvelynx.utils.extrapolate import ConstantPadding
from lightcurvelynx.utils.plotting import plot_lightcurves

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# set a seed
np.random.seed(190814)


## misc constants
SNR_MIN = 3 # minimum SNR requirement

## two-component kilonova model parameters from Villar+17
MEJ_1 = 0.023
VEJ_1 = 0.256
TEMPERATURE_FLOOR_1 = 3983
KAPPA_1 = 0.5
MEJ_2 = 0.05
VEJ_2 = 0.149
TEMPERATURE_FLOOR_2 = 1151
KAPPA_2 = 3.65

parameters = {"mej_1":MEJ_1,
              "vej_1":VEJ_1,
              "temperature_floor_1":TEMPERATURE_FLOOR_1,
              "kappa_1":KAPPA_1,
              "mej_2":MEJ_2,
              "vej_2":VEJ_2,
              "temperature_floor_2":TEMPERATURE_FLOOR_2,
              "kappa_2":KAPPA_2
}

## t0, ra, dec, and redshift of host galaxy, for a mock candidate
t0_241109_og = 60623.74959
# t0_241109 = 61345 # start of IR1 (1 Nov 2026)
t0_241109 = 62137 # start of O5 (1 Jan 2029)

t0_241109 += 464 # time of a Rubin-LSST visit
ra_241109 = 283.2916 # mock-241109bn
dec_241109 = -28.1980
z_241109_host = 0.063


## helper functions
def flux2mag(flux_njy, dflux_njy):
    mag = -2.5*np.log10(flux_njy * 1e-9) + 8.9
    dmag = 2.5 * dflux_njy*1e-9 / (flux_njy * 1e-9) / np.log(10)
    return mag, dmag


## redback
rb_model_name = "two_component_kilonova_model"
rb_model = model_library.all_models_dict[rb_model_name]
rb_model_metadata = model_library.get_model_metadata(
    rb_model_name,
    default=None,
)
print(f"redback model metadata: {rb_model_metadata}")


## sim Rubin-LSST lightcurves for mock candidates counterparts to GW241109_115924
# set filters
filters = ["g", "r", "i", "z"]

# load the OpSim data
print("Loading OpSim (Rubin) simulation data...")
opsim_db = OpSim.from_url(
    "https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/baseline_v5.3.0_10yrs.db"
)
# opsim_db = OpSim.from_db(
#     "~/.cache/lightcurvelynx/opsim/baseline_v5.3.0_10yrs.db"
# )
print(f"Loaded OpSim with {len(opsim_db)} rows.")

# filter to only include rows that match filters we want to sim
filter_mask = np.isin(opsim_db["filter"], filters)
opsim_db = opsim_db.filter_rows(filter_mask)
t_min, t_max = opsim_db.time_bounds()
print(f"Filtered OpSim to {len(opsim_db)} rows and times [{t_min}, {t_max}]")

# load the passband data for the filters only; survey info
table_dir = _LIGHTCURVELYNX_BASE_DATA_DIR / "passbands" / "LSST"
passband_group = PassbandGroup.from_preset(
    preset="LSST",
    filters=filters,
    units="nm",
    trim_quantile=0.001,
    delta_wave=1,
    table_dir=table_dir,
)
survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

# create the model itself
parameters["redshift"] = z_241109_host
source = RedbackWrapperModel(
    rb_model,
    parameters=parameters,  # Set ALL the redback model parameters
    ra=ra_241109,
    dec=dec_241109,
    t0=t0_241109,
    node_label="source",
    time_extrapolation=(ConstantPadding(0.0), ConstantPadding(0.0)),
    model_metadata=rb_model_metadata,
)

# simulate the lightcurves
lightcurves = simulate_lightcurves(
    source,
    100,
    survey_info,
    rest_time_window_offset=(-0.05, 10)
)

# helpful prints
for i in range(len(lightcurves)):
    print(lightcurves.loc[i]["lightcurve"])

## plot + store all mags and mag errors
mag_arr = []
dmag_arr = []
for idx in range(len(lightcurves)):
    lc = lightcurves.loc[idx]
    if lc["nobs"] == 0:
        continue

    # get t0, redshift for plot title
    t0 = lc["params"]["source.t0"]
    redshift = lc["params"]["source.redshift"]

    # unpack the nested columns (filters, mjd, flux, and flux error)
    lc_filters = np.asarray(lc["lightcurve"]["filter"], dtype=str)
    lc_mjd = np.asarray(lc["lightcurve"]["mjd"], dtype=float)
    lc_flux = np.asarray(lc["lightcurve"]["flux"], dtype=float)
    lc_fluxerr = np.asarray(lc["lightcurve"]["fluxerr"], dtype=float)

    # convert flux densities (in nJy) to mags
    lc_mag, lc_magerr = flux2mag(lc_flux, lc_fluxerr)
    
    # remove points with SNR < some number and negative magerr, for plotting
    mask_snr = np.log(10) / 2.5 / lc_magerr > SNR_MIN
    mask_pos = lc_magerr > 0
    mask = mask_snr & mask_pos
    lc_filters_masked = lc_filters[mask]
    lc_mjd_masked = lc_mjd[mask]
    lc_flux_masked = lc_flux[mask]
    lc_fluxerr_masked = lc_fluxerr[mask]
    lc_mag_mask = lc_mag[mask]
    lc_magerr_mask = lc_magerr[mask]

    # append to big array
    # when SNR too low or error is negative, change to nan
    lc_mag_toappend = lc_mag.copy()
    lc_mag_toappend[mask] = np.nan
    lc_magerr_toappend = lc_magerr.copy()
    lc_magerr_toappend[mask] = np.nan
    mag_arr.append(lc_mag_toappend)
    dmag_arr.append(lc_magerr_toappend)
    
    # make the plot
    fig = plt.figure(figsize=(6,4))

    # plot the light curves
    ax = plot_lightcurves(
        figure=fig,
        fluxes=lc_flux_masked,
        # times=lc_mjd,
        times=lc_mjd_masked-t0_241109,
        fluxerrs=lc_fluxerr_masked,
        filters=lc_filters_masked,
        plot_magnitudes=True,
        title=f"sample {idx}; t0={t0:.0f} MJD; redshift={redshift:.3f}",
    )
    # ax.plot([t0, t0], ax.get_ylim(), "k-", label="t0") 
    ax.plot([0, 0], ax.get_ylim(), "k-", label="t0") # if plottimg lc_mjd - t0_241109
    ax.set_xlabel("time [days]") # if plottimg lc_mjd - t0_241109
    ax.legend()
    plt.savefig(
        f"plots_GW241109_115924/mock_lightcurve_GW241109_115924_Rubin-LSST_{idx:03d}.png", 
        bbox_inches="tight")
    plt.close()

# make a dataframe of mean magnitudes and errors
med_mag_arr = np.nanmedian(mag_arr, axis=0)
med_dmag_arr = np.nanmedian(dmag_arr, axis=0)
filters_arr = lc_filters
mjds = lc_mjd - t0_241109 + t0_241109_og # need to subtract time offset to enable scoring
limits = [""]*len(mjds)
sources = ["Rubin"]*len(mjds)

df = pd.DataFrame(
    data=np.array([mjds, filters_arr, med_mag_arr, med_dmag_arr, limits, sources]).T,
    columns=["time", "filter", "magnitude", "error", "limit", "source"])

df.to_csv("tables_GW241109_115924/phot.csv", index=False)
