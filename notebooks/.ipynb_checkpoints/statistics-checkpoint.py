# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # statistics.ipynb
#
# This notebook evaluates the main results for H1 and H3 using a simple run-level approach.
#
# For each metric, it reports:
# - a Friedman test across the three conditions
# - the mean difference between each colored condition and the normal condition
#
# H1 metrics:
# - mean target CCA correlation
# - mean CCA margin
#
# H3 metrics:
# - accuracy
# - ITR
#
# Input:
# - `../derived/run_summary.csv`
#
# This notebook is designed for a simple single-participant analysis.

# %% [markdown]
# ### Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

# %% [markdown]
# ### Load inputs
# Use `run_summary.csv`

# %%
INPUT_DIR = Path("../derived")

# Load run-level summary values from the previous notebook
run_summary = pd.read_csv(INPUT_DIR / "run_summary.csv")

print("run_summary shape:", run_summary.shape)
run_summary.head()


# %% [markdown]
# ### Helper Functions
# - `get_metric_by_condition`
# - `run_friedman_test`
# - `mean_difference`

# %%
def get_metric_by_condition(df, metric_name, condition_name):
    """
    Return one metric column for one condition, sorted by run.

    Parameters
    ----------
    df : DataFrame
        Run-level summary table.
    metric_name : str
        Name of the metric column to extract.
    condition_name : str
        Name of the condition to filter.

    Returns
    -------
    values : ndarray
        Metric values for this condition, sorted by run.
    """
    subset = df[df["condition"] == condition_name].sort_values("run")
    return subset[metric_name].values


def run_friedman_test(df, metric_name):
    """
    Run a Friedman test across the three conditions.

    Parameters
    ----------
    df : DataFrame
        Run-level summary table.
    metric_name : str
        Name of the metric column to test.

    Returns
    -------
    statistic : float
        Friedman test statistic.
    p_value : float
        P-value from the Friedman test.
    """
    normal = get_metric_by_condition(df, metric_name, "normal")
    red_green = get_metric_by_condition(df, metric_name, "red_green")
    blue_yellow = get_metric_by_condition(df, metric_name, "blue_yellow")

    statistic, p_value = friedmanchisquare(normal, red_green, blue_yellow)
    return statistic, p_value


def mean_difference(df, metric_name, cond_a, cond_b):
    """
    Compute the mean paired difference: cond_b - cond_a.

    Positive values mean cond_b tends to be larger than cond_a.
    Negative values mean cond_b tends to be smaller than cond_a.
    """
    values_a = get_metric_by_condition(df, metric_name, cond_a)
    values_b = get_metric_by_condition(df, metric_name, cond_b)
    return float(np.mean(values_b - values_a))


# %% [markdown]
# ### Descriptive Summary

# %%
descriptive_summary = (
    run_summary
    .groupby("condition", as_index=False)
    .agg(
        mean_rho_target_mean=("mean_rho_target", "mean"),
        mean_rho_target_median=("mean_rho_target", "median"),
        mean_rho_margin_mean=("mean_rho_margin", "mean"),
        mean_rho_margin_median=("mean_rho_margin", "median"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_median=("accuracy", "median"),
        ITR_mean=("ITR", "mean"),
        ITR_median=("ITR", "median"),
    )
)

descriptive_summary

# %% [markdown]
# ### H1 Statistics

# %%
# H1 uses two run-level metrics:
# 1. mean target CCA correlation
# 2. mean CCA margin
h1_metrics = ["mean_rho_target", "mean_rho_margin"]

h1_rows = []

for metric in h1_metrics:
    # Test whether there is any overall difference across the three conditions
    friedman_stat, friedman_p = run_friedman_test(run_summary, metric)

    # Compute simple direction-of-effect values
    # Positive values mean the colored condition is higher than normal
    diff_red_green = mean_difference(run_summary, metric, "normal", "red_green")
    diff_blue_yellow = mean_difference(run_summary, metric, "normal", "blue_yellow")

    h1_rows.append({
        "hypothesis": "H1",
        "metric": metric,
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "mean_difference_red_green_minus_normal": diff_red_green,
        "mean_difference_blue_yellow_minus_normal": diff_blue_yellow,
    })

h1_stats = pd.DataFrame(h1_rows)
h1_stats

# %% [markdown]
# ### H2 Statistics

# %%
# H2 uses two run-level metrics:
# 1. accuracy
# 2. ITR
h3_metrics = ["accuracy", "ITR"]

h3_rows = []

for metric in h3_metrics:
    # Test whether there is any overall difference across the three conditions
    friedman_stat, friedman_p = run_friedman_test(run_summary, metric)

    # Compute simple direction-of-effect values
    # Positive values mean the colored condition is higher than normal
    diff_red_green = mean_difference(run_summary, metric, "normal", "red_green")
    diff_blue_yellow = mean_difference(run_summary, metric, "normal", "blue_yellow")

    h3_rows.append({
        "hypothesis": "H3",
        "metric": metric,
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "mean_difference_red_green_minus_normal": diff_red_green,
        "mean_difference_blue_yellow_minus_normal": diff_blue_yellow,
    })

h3_stats = pd.DataFrame(h3_rows)
h3_stats

# %% [markdown]
# ### Combine all results

# %%
# Combine H1 and H3 results into one table
all_stats = pd.concat([h1_stats, h3_stats], ignore_index=True)

all_stats

# %% [markdown]
# ### Print interpretations

# %%
alpha = 0.05

for _, row in all_stats.iterrows():
    print(f"--- {row['hypothesis']} | {row['metric']} ---")

    # Interpret the overall Friedman test
    if row["friedman_p_value"] < alpha:
        print(f"Overall condition difference: significant (p = {row['friedman_p_value']:.4f})")
    else:
        print(f"Overall condition difference: not significant (p = {row['friedman_p_value']:.4f})")

    # Interpret the red-green difference
    diff_rg = row["mean_difference_red_green_minus_normal"]
    if diff_rg > 0:
        print(f"Red-Green is higher than Normal on average (mean difference = {diff_rg:.4f})")
    elif diff_rg < 0:
        print(f"Red-Green is lower than Normal on average (mean difference = {diff_rg:.4f})")
    else:
        print(f"Red-Green and Normal are equal on average (mean difference = {diff_rg:.4f})")

    # Interpret the blue-yellow difference
    diff_by = row["mean_difference_blue_yellow_minus_normal"]
    if diff_by > 0:
        print(f"Blue-Yellow is higher than Normal on average (mean difference = {diff_by:.4f})")
    elif diff_by < 0:
        print(f"Blue-Yellow is lower than Normal on average (mean difference = {diff_by:.4f})")
    else:
        print(f"Blue-Yellow and Normal are equal on average (mean difference = {diff_by:.4f})")

    print()

# %%
decision_rows = []

for _, row in all_stats.iterrows():
    supports_red_green = row["mean_difference_red_green_minus_normal"] > 0
    supports_blue_yellow = row["mean_difference_blue_yellow_minus_normal"] > 0

    decision_rows.append({
        "hypothesis": row["hypothesis"],
        "metric": row["metric"],
        "friedman_p_value": row["friedman_p_value"],
        "red_green_higher_than_normal": supports_red_green,
        "blue_yellow_higher_than_normal": supports_blue_yellow,
    })

decision_table = pd.DataFrame(decision_rows)
decision_table

# %%
