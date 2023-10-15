"""
Plot time-frequency responses contra- vs. ipsilateral to previous button-press
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory

from tfrplot import contrast_tfr_plots

memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"], verbose=0)

c_prev_resp = pd.read_hdf("Figure 3-Source Data 1.hdf")
contrast_tfr_plots.plot_streams_fig(
    c_prev_resp,
    contrast_name="prev_hand",
    configuration=contrast_tfr_plots.example_config,
    stats=True,
)
plt.suptitle("Contra- vs. ipsilateral to previous button-press")
plt.show()
