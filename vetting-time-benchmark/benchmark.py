import os, time, json
import numpy as np
import pandas as pd
from trove_targets.models import Target
from tom_nonlocalizedevents.models import EventCandidate
from scoring.scoring import (
    get_distance_score,
    host_distance_match,
    skymap_association,
    clean_host_df,
    _minor_planet_score
)
from candidate_vetting.vet import (
    point_source_association,
    agn_association_2d,
    host_association
)
from scoring.kilonova_scorer_helpers import util as knscorer_util
from scoring.em_models.util import MODELS_BY_NAME 
from scoring.vet_phot import (
    _get_post_disc_phot,
    _score_phot,
    _get_pre_disc_phot,
    get_predetection_stats,
    PHOT_SCORE_MIN,
    PREDETECTION_SNR_THRESHOLD,
)

TEST_NLE_ID = 20113
_m = MODELS_BY_NAME["KN"]
PARAM_RANGES = _m.phot_parameter.model_dump() | {"t_post": _m.t_post, "t_pre": _m.t_pre}

def timeit(f, *args, **kwargs):
    start = time.time()
    r = f(*args, **kwargs)
    end = time.time()
    return end-start, r

def vetting_wrapper(ec):

    nonlocalized_event_name = ec.nonlocalizedevent.event_id
    nonlocalized_event = ec.nonlocalizedevent
    target = ec.target
    target_id = ec.target_id
    
    result = {}
    
    # skymap score
    result["2D"],_ = timeit(
        skymap_association,
        nonlocalized_event_name,
        target_id
    )

    # PS
    result["ps_association"],_ = timeit(
        point_source_association,
        target_id
    )
    
    # MPC
    result["MPC"],_ = timeit(
        _minor_planet_score,
        ec.target,
        ec.target.basetarget_ptr.targetextra_set.all()
    )
    
    # AGN association
    result["agn_association"],_ = timeit(
        agn_association_2d,
        target_id
    )
    
    # host association
    result["host_association"],host_df = timeit(
        host_association,
        target_id
    )
    
    # distance score
    host_df = clean_host_df(host_df)
    result["distance_score"], _ = timeit(
        _distance_score_wrapper,
        host_df,
        target_id,
        nonlocalized_event_name
    )

    # KNScorer
    try:
        knscorer_time,_ = timeit(
            knscorer_util.kilonova_score_candidate,
            target_id,
            ec.nonlocalizedevent
        )
    except knscorer_util.KilonovaScoreUnavailable:
        knscorer_time = None
    result["phot_kn_scorer"] =  knscorer_time

    # metric based photometry scorer
    result["phot_metric_based"],_ = timeit(
        _phot_score_wrapper,
        target,
        ec.nonlocalizedevent
    )

    result["ec_id"] = ec.id
    
    return result

def _phot_score_wrapper(target, nonlocalized_event, param_ranges=PARAM_RANGES):
    
    allphot = _get_post_disc_phot(
        target_id=target.id,
        nonlocalized_event=nonlocalized_event,
        t_post=param_ranges["t_post"],
    )
    phot_score, lum, max_time, decay_rate, _, _ = _score_phot(
        allphot=allphot,
        target=target,
        nonlocalized_event=nonlocalized_event,
        param_ranges=param_ranges,
        filt=[
            "g",
            "r",
            "i",
            "z",
            "F129",
            "F158",
            "o",
            "c",
        ],  # common optical filters + some Roman filters + ATLAS o,c
    )

    prephot = _get_pre_disc_phot(
        target_id=target.id,
        nonlocalized_event=nonlocalized_event,
        t_pre=param_ranges["t_pre"],
    )
    predet_score = 1
    if prephot is not None and len(prephot):
        try:
            n_predets, _ = get_predetection_stats(
                prephot.mjd.values,
                prephot.magerr.values,
                window_size=5,  # +/-5 day window size
                det_snr_thresh=PREDETECTION_SNR_THRESHOLD,
            )
        except ValueError:
            n_predets = [
                0
            ]  # this ValueError only happens when there aren't any predets
        if any(v >= param_ranges["max_predets"] for v in n_predets):
            predet_score = PHOT_SCORE_MIN

    
def _distance_score_wrapper(host_df, target_id, nonlocalized_event_name, nle_type="SSM"):
    target = Target.objects.get(id=target_id)
    if target.redshift is not None and not np.isnan(target.redshift):
        # use target redshift, so no need to compute distance scores for galaxies
        host_score, host_name = get_distance_score(
            host_df, target_id, nonlocalized_event_name
        )
        
    elif len(host_df) != 0:
        # then run the distance comparison for each of these hosts
        host_df = host_distance_match(host_df, target_id, nonlocalized_event_name)
        
        # choose the maximum score
        if nle_type not in {"FXT", "LGRB", "SGRB"}: # These don't have distances
            host_score, host_name, host_catalog = get_distance_score(
                host_df, target_id, nonlocalized_event_name
            )
        
    else:
        # if no target redshift is known and no hosts are found, we don't want
        # to bias the final score (host may just be too far)
        score = 1

def benchmark(
        result_file=os.path.join(os.getcwd(), "time-benchmark-result.json"),
        overwrite=False,
        nle_id=TEST_NLE_ID
):

    result = []
    if os.path.exists(result_file) and not overwrite:
        with open(result_file, "r") as f:
            result+=json.load(f)

    ecids = {d["ec_id"] for d in result}
    
    for ec in EventCandidate.objects.filter(nonlocalizedevent_id=nle_id):
        print(f"Vetting EventCandidate({ec.id}): {ec.target.name} associated with {ec.nonlocalizedevent.event_id}")
        if ec.id in ecids and not overwrite:
            print(f"Skipping {ec.id}, already in the result file!")
            continue # we don't need to rerun

        vet_total_time, vet_result = timeit(
            vetting_wrapper,
            ec
        )
        vet_result["total_time"] = vet_total_time
        
        # this is kinda slow, but maybe it's fine
        result.append(vet_result)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4)

    return result
