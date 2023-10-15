"""
Plot time course of single-trial history bias estimates for one example participant
and block from the Neutral condition
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Figure 1-Source Data 5.csv", sep="\t")

max_val = max(df.single_trial_bias_neutral.values)
min_val = min(df.single_trial_bias_neutral.values)
bin_size = (max_val - min_val) / 3
plt.fill_between(
    range(len(df.values)),
    (min_val + 2 * bin_size),
    (min_val + bin_size),
    color="lightgrey",
)
plt.plot(df.single_trial_bias_neutral, "purple")
plt.ylabel("Bias")
plt.xlabel("Trial #")
plt.suptitle("Model-derived bias time course (example)")
plt.show()
