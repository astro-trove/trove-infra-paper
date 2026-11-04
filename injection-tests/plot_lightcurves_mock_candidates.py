#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. Created on Fri Oct  2 15:36:36 2026
.. @author: Nicholas Vieira
.. @plot_lightcurves_mock_candidates.py

"""

import pandas as pd

from tom_nonlocalizedevents.models import NonLocalizedEvent 

from trove_targets.models import Target 
from scoring.vet_phot import _get_post_disc_phot, _get_pre_disc_phot

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from matplotlib.ticker import FormatStrFormatter

## constants: plotting
BAND_COLOR_MAPPING = {'g':'#01ff07', # bright green
                      'r':'#ff474c', # bright red
                      'i':'#fa5ff7', # light magenta
                      'z':'#ff9408',
}
FACILITY_MARKER_MAPPING = {'CTIO':'s',
                           'Rubin':'o',
                           'ZTF':'d',
}

FACILITY_MARKERSIZE_MAPPING = {'CTIO':12.0,
                               'Rubin':12.0,
                               'ZTF':12.0,
}

YLIMS_PER_TARGET = [(21.9, 20.3),
                    (23.75, 22.35),
                    (26.1, 22.2)]

## NLEs of interest
NLE_NAMES = ["S190425z",
             "S250206cm",
             "S241109bn",
]
NLE_NAMES_NICE = ["GW190425",
                  "S250206dm",
                  "GW241109_115924",
]
NLE_IDS = ["22106",
           "19294",
           "16334",
]

## targets of interest
TARGET_NAMES = ["mock-190425z",
                "mock-250206dm",
                "mock-241109bn"
]
TARGET_SCORES_NICE = [r"$S_{\mathrm{KN}} = 0.71{\ }\mathrm{or}{\ }0.53$",
                      r"$S_{\mathrm{KN}} = 0.46{\ }\mathrm{or}{\ }0.33$",
                      r"$S_{\mathrm{KN}} = 0.24{\ }\mathrm{or}{\ }0.16$",
]

## set up targets, NLEs
targets = [Target.objects.get(name=n) for n in TARGET_NAMES]
nles = [NonLocalizedEvent.objects.get(id=nid) for nid in NLE_IDS] 


## for collecting all photometry, set up a dummy dataframe with no data
allphot_allcands = _get_pre_disc_phot(target_id=targets[0].id, 
                                      nonlocalized_event=nles[0],
                                      t_pre=0)[:0]
allphot_allcands["name"] = []
allphot_allcands.set_index("name")


## set up figure
fig, axs = plt.subplots(3, 1, figsize=(12,12),
                        constrained_layout=True)                    
axs_flat = axs.flatten()

# dummy points/bands, for legend
for facility in list(FACILITY_MARKER_MAPPING.keys()):
    if facility == "CTIO":
        facility_nice = "CTIO-DECam"
    elif facility == "Rubin":
        facility_nice = "Rubin-LSST"
    else:
        facility_nice = facility
    axs[0].plot([-100], [21], 
                marker=FACILITY_MARKER_MAPPING[facility], 
                ms=FACILITY_MARKERSIZE_MAPPING[facility],
                mew=3.0, mec="k", 
                mfc="white", color="k",
                label=facility_nice,
                lw=0)

for key, value in BAND_COLOR_MAPPING.items():
    plt.fill_between(x=[-100,-80], y1=[21,21], y2=[22,22], 
                     color=BAND_COLOR_MAPPING[key],
                     label=r"$"+key+r"$")
    


## iterate through targets
for i in range(len(targets)):
    print(f"\n\nTARGET: {targets[i]}") 
    
    # pre-GW photometry
    prephot = _get_pre_disc_phot(target_id=targets[i].id, 
                                 nonlocalized_event=nles[i],
                                 t_pre=0)
    
    # post-GW photometry
    postphot = _get_post_disc_phot(target_id=targets[i].id, 
                                  nonlocalized_event=nles[i],
                                  t_post=15)
    
    # merge 
    allphot = pd.concat([prephot, postphot]).sort_values(by="dt")
    print(allphot) # print photometry
    
    # for merging with dataframe collecting all candidates' photometry
    allphot["name"] = [targets[i].name]*len(allphot)

    # # toss out (1) upper limits and (2) low SNR
    # # (this is normally done within _score_phot)
    # allphot = allphot[~allphot.upperlimit]
    # allphot = allphot[allphot.snr >= 5]
    print(allphot) # print again
    
    # merge with big dataframe
    allphot_allcands = pd.concat([allphot_allcands, allphot])
    
    for _, allp in allphot.iterrows():
        if "CTIO" in allp.telescope:
            facility = "CTIO"
        elif "Rubin" in allp.telescope:
            facility = "Rubin"
        elif "ZTF" in allp.telescope:
            facility = "ZTF"
        else:
            raise ValueError(f"Did not recognize telescope {allp.telescope}")
            
        axs_flat[i].errorbar(allp["dt"], allp["mag"], yerr=allp["magerr"], 
                        marker=FACILITY_MARKER_MAPPING[facility], 
                        ms=FACILITY_MARKERSIZE_MAPPING[facility],  
                        mfc=BAND_COLOR_MAPPING[allp["filter"]],
                        color=BAND_COLOR_MAPPING[allp["filter"]],
                        mew=3.0, mec="k",
                        # label=facility,
                        lw=2.0, 
                        capsize=6.0)

    # reverse y-axis
    axs_flat[i].invert_yaxis()
    
    # # mark t=0
    # axs_flat[i].axvline(0, lw=2.0, ls="--", color="k")
    
    # pretty plotting
    axs_flat[i].grid()
    axs_flat[i].set_xlim(0-1e-10, 7)
    axs_flat[i].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if i == 2:
        axs_flat[i].set_xlabel("time since merger [days]", fontsize=36)
        axs_flat[i].tick_params(axis="both", which="both", labelsize=36, size=6)
    else:
        axs_flat[i].tick_params(axis="x", which="both", labelsize=1, size=6)
        axs_flat[i].tick_params(axis="y", which="both", labelsize=36, size=6)
        
    if i == 1:
        axs_flat[i].set_ylabel("apparent magnitude (AB)", fontsize=36)
        
    # label candidate
    bbox_props = dict(boxstyle='round', facecolor='white', edgecolor="k")
    text_x = 0.98
    ha = "right"
    text_y = 0.935
    va = "top"
    axs_flat[i].text(text_x, text_y, 
                     s=f"{NLE_NAMES_NICE[i]} mock candidate\n{TARGET_SCORES_NICE[i]}",
                     transform=axs_flat[i].transAxes,
                     horizontalalignment=ha,
                     verticalalignment=va,
                     fontsize=28,
                     bbox=bbox_props)
    axs_flat[i].set_ylim(YLIMS_PER_TARGET[i])
    

fig.legend(fontsize=28, loc="outside upper center", fancybox=True,
           bbox_to_anchor=(0.5, 1.17),
           ncol=3)

plt.savefig("./plots_lightcurves/lightcurves_mock_candidates.png", 
            bbox_inches="tight")
plt.savefig("./plots_lightcurves/lightcurves_mock_candidates.pdf", 
            bbox_inches="tight")
