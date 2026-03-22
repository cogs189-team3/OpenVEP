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
# # visualization.ipynb
#
# This notebook visualizes the main results for H1 and H2.
#
# For H1, it plots:
# - mean target CCA correlation by condition
# - mean CCA margin by condition
#
# For H2 (former H3), it plots:
# - run-level accuracy by condition
# - run-level ITR by condition
#
# Inputs:
# - `../derived/trial_results.csv`
# - `../derived/run_summary.csv`
#
# This notebook does not recompute features or classification.
# It only reads saved results and makes figures.

# %%
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ### Basic settings

# %%
# Folder that contains the saved CSV files
INPUT_DIR = Path("../derived")

# Folder where figures will be saved
OUTPUT_DIR = Path("../figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed order for conditions
# Using a fixed order makes all plots easier to compare
CONDITION_ORDER = ["normal", "red_green", "blue_yellow"]

# Use a clean plotting style
sns.set_theme(style="whitegrid")

# %% [markdown]
# ### Load inputs

# %%
# Load trial-level results
trial_results = pd.read_csv(INPUT_DIR / "trial_results.csv")

# Load run-level summary values
run_summary = pd.read_csv(INPUT_DIR / "run_summary.csv")

print("trial_results shape:", trial_results.shape)
print("run_summary shape:", run_summary.shape)

run_summary.head()

# %% [markdown]
# #### Make condition labels easier to read

# %%
condition_label_map = {
    "normal": "Normal",
    "red_green": "Red-Green",
    "blue_yellow": "Blue-Yellow",
}

# Add a new column with display labels
run_summary["condition_label"] = run_summary["condition"].map(condition_label_map)

# Keep the plotting order fixed
plot_order = [condition_label_map[c] for c in CONDITION_ORDER]

run_summary.head()

# %% [markdown]
# ### Plot H1: mean target CCA correlation

# %%
# Create a figure for H1 feature 1:
# mean target CCA correlation for each run in each condition
plt.figure(figsize=(8, 5))

# Show each run as an individual point
sns.stripplot(
    data=run_summary,
    x="condition_label",
    y="mean_rho_target",
    order=plot_order,
    size=8
)

# Show the condition mean as a black point connected by lines
sns.pointplot(
    data=run_summary,
    x="condition_label",
    y="mean_rho_target",
    order=plot_order,
    errorbar=None,
    color="black"
)

# Add labels and title
plt.xlabel("Condition")
plt.ylabel("Mean target CCA correlation")
plt.title("H1: Mean target CCA correlation by condition")

# Save the figure
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "h1_mean_rho_target.png", dpi=300)

# Show the figure
plt.show()

# %% [markdown]
# ### Plot H1: mean CCA margin

# %%
# Create a figure for H1 feature 2:
# mean CCA margin for each run in each condition
plt.figure(figsize=(8, 5))

# Show each run as an individual point
sns.stripplot(
    data=run_summary,
    x="condition_label",
    y="mean_rho_margin",
    order=plot_order,
    size=8
)

# Show the condition mean as a black point connected by lines
sns.pointplot(
    data=run_summary,
    x="condition_label",
    y="mean_rho_margin",
    order=plot_order,
    errorbar=None,
    color="black"
)

# Add labels and title
plt.xlabel("Condition")
plt.ylabel("Mean CCA margin")
plt.title("H1: Mean CCA margin by condition")

# Save the figure
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "h1_mean_rho_margin.png", dpi=300)

# Show the figure
plt.show()

# %% [markdown]
# ### Plot H3: accuracy

# %%
# Create a figure for H2 feature 1:
# classification accuracy for each run in each condition
plt.figure(figsize=(8, 5))

# Show each run as an individual point
sns.stripplot(
    data=run_summary,
    x="condition_label",
    y="accuracy",
    order=plot_order,
    size=8
)

# Show the condition mean as a black point connected by lines
sns.pointplot(
    data=run_summary,
    x="condition_label",
    y="accuracy",
    order=plot_order,
    errorbar=None,
    color="black"
)

# Add labels and title
plt.xlabel("Condition")
plt.ylabel("Accuracy")
plt.title("H2: Run-level accuracy by condition")

# Limit the y-axis to the natural range of accuracy
plt.ylim(0, 1)

# Save the figure
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "h3_accuracy.png", dpi=300)

# Show the figure
plt.show()

# %% [markdown]
# ### Plot H2: ITR

# %%
# Create a figure for H2 feature 2:
# Information Transfer Rate (ITR) for each run in each condition
plt.figure(figsize=(8, 5))

# Show each run as an individual point
sns.stripplot(
    data=run_summary,
    x="condition_label",
    y="ITR",
    order=plot_order,
    size=8
)

# Show the condition mean as a black point connected by lines
sns.pointplot(
    data=run_summary,
    x="condition_label",
    y="ITR",
    order=plot_order,
    errorbar=None,
    color="black"
)

# Add labels and title
plt.xlabel("Condition")
plt.ylabel("ITR (bits/min)")
plt.title("H2: Run-level ITR by condition")

# Save the figure
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "h3_itr.png", dpi=300)

# Show the figure
plt.show()

# %% [markdown]
# ### Optional Test: Relationship between H1 and H2
# Does stronger target correlation tend to go with higher accuracy?

# %%
plot_order = ["Normal", "Red-Green", "Blue-Yellow"]
palette = {
    "Normal": "#dd8452",
    "Red-Green": "#55a868",
    "Blue-Yellow": "#4c72b0"
}

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=run_summary,
    x="mean_rho_target",
    y="accuracy",
    hue="condition_label",
    hue_order=plot_order,
    palette=palette,
    s=80
)

for label in plot_order:
    subset = run_summary[run_summary["condition_label"] == label]

    sns.regplot(
        data=subset,
        x="mean_rho_target",
        y="accuracy",
        scatter=False,
        ci=None,
        color=palette[label]   # ← condition と同じ色を明示
    )

plt.xlabel("Mean target CCA correlation")
plt.ylabel("Accuracy")
plt.title("Relationship between H1 and H3 measures by condition")
plt.tight_layout()
plt.show()

# %%
from scipy.stats import pearsonr

for label in plot_order:
    subset = run_summary[run_summary["condition_label"] == label]

    r, p = pearsonr(subset["mean_rho_target"], subset["accuracy"])

    print(f"{label}: r = {r:.3f}, p = {p:.3f}")

# %%
from scipy.stats import pearsonr, spearmanr

for label in plot_order:
    subset = run_summary[run_summary["condition_label"] == label]

    x = subset["mean_rho_target"]
    y = subset["accuracy"]

    r_pearson, p_pearson = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)

    print(f"{label}")
    print(f"  Pearson  r = {r_pearson:.3f}, p = {p_pearson:.3f}")
    print(f"  Spearman r = {r_spearman:.3f}, p = {p_spearman:.3f}")

# %% [markdown]
# ### Sumary

# %%
summary_table = (
    run_summary
    .groupby("condition_label", as_index=False)
    .agg(
        mean_rho_target=("mean_rho_target", "mean"),
        mean_rho_margin=("mean_rho_margin", "mean"),
        accuracy=("accuracy", "mean"),
        ITR=("ITR", "mean")
    )
)

summary_table

# %%
