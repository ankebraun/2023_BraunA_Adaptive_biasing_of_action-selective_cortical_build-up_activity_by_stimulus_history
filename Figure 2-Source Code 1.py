"""
Plot time-frequency responses of Neural signatures of stimulus processing and
action planning across the cortical visuo-motor pathway
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory

from tfrplot import contrast_tfr_plots

memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"], verbose=0)

c_stim_all = pd.read_hdf("Figure 2-Source Data 1.hdf")
c_stim_lo_coh = pd.read_hdf("Figure 2-Source Data 2.hdf")
c_stim_hi_coh = pd.read_hdf("Figure 2-Source Data 3.hdf")
c_resp_no_epoch = pd.read_hdf("Figure 2-Source Data 4.hdf")

# Overall task-related power change averaged across hemispheres
contrast_tfr_plots.plot_streams_fig(
    c_stim_all.query('epoch != "response"'),
    contrast_name="all",
    configuration=contrast_tfr_plots.example_config,
    stats=True,
)
plt.suptitle("Overall task-related power change")

c_stim_diff_hi_lo = pd.DataFrame(
    c_stim_hi_coh.to_numpy() - c_stim_lo_coh.to_numpy(),
    index=c_stim_hi_coh.index,
    columns=c_stim_hi_coh.columns,
)
# Difference in time-frequency response between high and low motion coherence average
# across hemispheres
contrast_tfr_plots.plot_streams_fig(
    c_stim_diff_hi_lo,
    contrast_name="0.81_coh",
    configuration=contrast_tfr_plots.example_config,
    stats=True,
)
plt.suptitle("Strong vs. weak motion coherence")

# Time-frequency representation of action-selective power lateralization contralateral
# vs. ipsilateral to upcoming button-press
contrast_tfr_plots.plot_streams_fig(
    c_resp_no_epoch,
    contrast_name="hand",
    configuration=contrast_tfr_plots.example_config,
    stats=True,
)
plt.suptitle("Contra- vs. ipsilateral button-press")
plt.show()
