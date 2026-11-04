"""
Based in part on https://github.com/nikhil-sarin/redback/blob/master/examples/simulate_kilonova.py
And https://github.com/nikhil-sarin/redback/blob/master/examples/simulate_single_transient_in_ztf.py
"""

import numpy as np

import redback
from redback.simulate_transients import SimulateOpticalTransient

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

## t0, ra, dec, and redshift of host galaxy, for a mock candidate
t0_250206 = 60713.14271
ra_250206 = 248.490000 # mock-250206dm
dec_250206 = -70.042000
z_250206_host = 0.073


## set up telescope pointings 
# number of observations, cadence, limiting mags in each filter
num_obs = {'desr': 20, 'desi':20}
average_cadence = {'desr': 1.0, 'desi': 1.0}
cadence_scatter = {'desr': 0.5, 'desi':0.5}
limiting_magnitudes = {'desr': 22.6, 'desi':23.1} # see GCN 39318

# use redback to make pointings table
pointings = redback.simulate_transients.make_pointing_table_from_average_cadence(
    ra=ra_250206, 
    dec=dec_250206, 
    num_obs=num_obs, 
    average_cadence=average_cadence,
    cadence_scatter=cadence_scatter, 
    limiting_magnitudes=limiting_magnitudes, 
    model_kwargs = {},
    initMJD=t0_250206
)
print(f"POINTINGS = {pointings}")


## set up parameters
model_kwargs = {}
parameters = redback.priors.get_priors(model='two_component_kilonova_model').sample()
parameters["mej_1"] = MEJ_1
parameters["vej_1"] = VEJ_1
parameters["temperature_floor_1"] = TEMPERATURE_FLOOR_1
parameters["kappa_1"] = KAPPA_1
parameters["mej_2"] = MEJ_2
parameters["vej_2"] = VEJ_2
parameters["temperature_floor_2"] = TEMPERATURE_FLOOR_2
parameters["kappa_2"] = KAPPA_2
parameters["t0"] = t0_250206
parameters["redshift"] = z_250206_host

# place the transient on the sky
parameters['ra'] = ra_250206
parameters['dec'] = dec_250206

## simulate
kn_sim = SimulateOpticalTransient.simulate_transient(
    model='two_component_kilonova_model',
    parameters=parameters, 
    model_kwargs=model_kwargs,
    pointings_database=pointings,
    survey=None,
    end_transient_time=14., 
    snr_threshold=SNR_MIN
)
print(f"OBSERVATIONS = {kn_sim.observations}")

# save simulated observations to a file
kn_sim.save_transient(name='S250206dm_mock_kilonova')

# load in data
kn_object = redback.transient.Kilonova.from_simulated_optical_data(
    name='S250206dm_mock_kilonova', 
    data_mode='magnitude')
kn_object.plot_data()
