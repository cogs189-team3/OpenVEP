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
# # feature_extraction_and_classification.ipynb
#
# This notebook does the following:
#
# 1. Load trial metadata and EEG data from the stimulus window
# 2. Create sinusoidal reference signals for all 32 SSVEP classes
# 3. Compute CCA scores between each EEG trial and each reference signal
# 4. Extract trial-level features for H1
# 5. Perform CCA-based classification for H3
# 6. Create run-level summary values
# 7. Save results for later visualization and statistics
#
# Outputs:
# - `../derived/trial_results.csv`
# - `../derived/run_summary.csv`

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

# %% [markdown]
# ### Basic Setting

# %%
# Input and output folders
INPUT_DIR = Path("../derived")
OUTPUT_DIR = Path("../derived")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Basic experiment settings
FS = 250                    # Sampling rate in Hz
N_CLASSES = 32              # Total number of SSVEP classes
N_CHANNELS = 8              # Number of EEG channels
N_SAMPLES_STIM = 300        # Number of samples in the stimulus window
TRIAL_DURATION = N_SAMPLES_STIM / FS   # 300 / 250 = 1.2 seconds

# Number of harmonics used to build reference signals
N_HARMONICS = 2

# (Keeping this small makes the analysis easier)

# %% [markdown]
# ### Load inputs

# %%
# Load metadata created in load_and_qc.ipynb
trial_metadata = pd.read_csv(INPUT_DIR / "trial_metadata.csv")

# Load EEG data from the stimulus window only
# Expected shape: (n_trials, 8, 300)
all_trials_stim = np.load(INPUT_DIR / "all_trials_stim.npy")

print("trial_metadata shape:", trial_metadata.shape)
print("all_trials_stim shape:", all_trials_stim.shape)

trial_metadata.head()

# %% [markdown]
# ### Define frequencies and phases
# * 32 classes (keys) = 8 frequencies × 4 phases

# %%
# The 32 classes are defined by:
# - 8 frequencies: 8, 9, ..., 15 Hz
# - 4 phase offsets
freqs = np.arange(8.0, 16.0, 1.0)
phases = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])

# Build a list of (frequency, phase) pairs
# Each pair corresponds to one class
class_params = []

for freq in freqs:
    for phase in phases:
        class_params.append((freq, phase))

# Make sure we really created 32 classes
assert len(class_params) == N_CLASSES

# Show the first few class definitions
class_params[:5]


# %% [markdown]
# ### Helper Functions
# 1. `make_reference_signal`
# 2. `cca_score`
# 3. `compute_itr`

# %%
def make_reference_signal(freq, phase, n_samples, fs, n_harmonics=2):
    """
    Create a sinusoidal reference signal for one SSVEP class.

    Parameters
    ----------
    freq : float
        Target frequency for this class.
    phase : float
        Target phase for this class.
    n_samples : int
        Number of time samples in the trial.
    fs : int
        Sampling rate in Hz.
    n_harmonics : int
        Number of harmonics to include.

    Returns
    -------
    ref_signal : ndarray of shape (n_samples, 2 * n_harmonics)
        Columns are sine/cosine components:
        [sin(1f), cos(1f), sin(2f), cos(2f), ...]
    """
    # Create a time vector in seconds
    t = np.arange(n_samples) / fs

    # Store all sine/cosine components here
    ref_components = []

    # Add one sine and one cosine for each harmonic
    for h in range(1, n_harmonics + 1):
        ref_components.append(np.sin(2 * np.pi * h * freq * t + h * phase))
        ref_components.append(np.cos(2 * np.pi * h * freq * t + h * phase))

    # Stack all components into a 2D array:
    # rows = time points, columns = reference components
    ref_signal = np.column_stack(ref_components)
    return ref_signal


def cca_score(eeg_trial, ref_signal):
    """
    Compute the first canonical correlation between one EEG trial
    and one reference signal.

    Parameters
    ----------
    eeg_trial : ndarray of shape (n_channels, n_samples)
        EEG data for one trial.
    ref_signal : ndarray of shape (n_samples, n_ref_components)
        Sinusoidal reference signal for one class.

    Returns
    -------
    r : float
        The first canonical correlation coefficient.
    """
    # CCA in sklearn expects shape: (n_samples, n_features)
    # EEG is currently (channels, samples), so transpose it
    X = eeg_trial.T
    Y = ref_signal

    # We only need the first canonical component
    cca = CCA(n_components=1)

    # Fit CCA to EEG and reference
    cca.fit(X, Y)

    # Transform both datasets into the canonical space
    X_c, Y_c = cca.transform(X, Y)

    # Compute correlation between the first canonical components
    r = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]

    # If numerical issues produce NaN, replace with 0
    if np.isnan(r):
        return 0.0

    return float(r)


def compute_itr(accuracy, n_classes, trial_duration):
    """
    Compute Information Transfer Rate (ITR) in bits/min.

    Parameters
    ----------
    accuracy : float
        Classification accuracy between 0 and 1.
    n_classes : int
        Number of possible classes.
    trial_duration : float
        Time per trial in seconds.

    Returns
    -------
    itr : float
        Information Transfer Rate in bits/min.
    """
    # Special case: accuracy = 0
    if accuracy <= 0:
        return 0.0

    # Special case: accuracy = 1
    if accuracy >= 1:
        return np.log2(n_classes) * 60.0 / trial_duration

    # Standard ITR formula
    term1 = np.log2(n_classes)
    term2 = accuracy * np.log2(accuracy)
    term3 = (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1))

    itr = (term1 + term2 + term3) * 60.0 / trial_duration
    return itr


# %% [markdown]
# ### Build reference signals for all classes

# %%
reference_signals = []

for freq, phase in class_params:
    ref_signal = make_reference_signal(
        freq=freq,
        phase=phase,
        n_samples=N_SAMPLES_STIM,
        fs=FS,
        n_harmonics=N_HARMONICS
    )
    reference_signals.append(ref_signal)

print("Number of reference signals:", len(reference_signals))
print("Shape of one reference signal:", reference_signals[0].shape)

# %% [markdown]
# ### Compute trial-level CCA feature and predictions

# %%
# Store one result row per trial
trial_result_rows = []

# Loop through all trials
for i in range(len(trial_metadata)):
    # Get EEG data for this trial
    # Shape: (8 channels, 300 samples)
    eeg_trial = all_trials_stim[i]

    # Get the true class label for this trial
    true_label = int(trial_metadata.loc[i, "true_label"])

    # Compute one CCA score for each of the 32 classes
    rho_all = []
    for ref_signal in reference_signals:
        rho = cca_score(eeg_trial, ref_signal)
        rho_all.append(rho)

    # Convert the list to a numpy array for easier indexing
    rho_all = np.array(rho_all)

    # Predicted label = the class with the highest CCA score
    predicted_label = int(np.argmax(rho_all))

    # H1 feature 1:
    # CCA score for the correct class
    rho_target = float(rho_all[true_label])

    # Highest CCA score among all classes
    rho_max = float(np.max(rho_all))

    # H1 feature 2:
    # Compare the correct class to the best incorrect class
    other_mask = np.ones(len(rho_all), dtype=bool)
    other_mask[true_label] = False
    rho_best_other = float(np.max(rho_all[other_mask]))
    rho_margin = rho_target - rho_best_other

    # H3 feature:
    # Was the predicted class correct?
    correct = int(predicted_label == true_label)

    # Save all important information for this trial
    trial_result_rows.append({
        "participant": int(trial_metadata.loc[i, "participant"]),
        "condition": trial_metadata.loc[i, "condition"],
        "run": int(trial_metadata.loc[i, "run"]),
        "trial_repeat": int(trial_metadata.loc[i, "trial_repeat"]),
        "class_id": int(trial_metadata.loc[i, "class_id"]),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": correct,
        "rho_target": rho_target,
        "rho_max": rho_max,
        "rho_margin": rho_margin,
    })

# %% [markdown]
# ### Create trial-level results table

# %%
# Convert the list of dictionaries into a DataFrame
trial_results = pd.DataFrame(trial_result_rows)

# Sort rows to keep the output easy to read
trial_results = trial_results.sort_values(
    by=["condition", "run", "trial_repeat", "class_id"]
).reset_index(drop=True)

print("trial_results shape:", trial_results.shape)
trial_results.head()

# %% [markdown]
# ### Create run-level summary

# %%
# Summarize trial-level values at the run level
# This is useful because later statistics will be done at the run level
run_summary = (
    trial_results
    .groupby(["condition", "run"], as_index=False)
    .agg(
        # H1 summaries
        mean_rho_target=("rho_target", "mean"),
        mean_rho_margin=("rho_margin", "mean"),

        # H3 summary
        accuracy=("correct", "mean"),

        # Number of trials in each run
        n_trials=("correct", "size"),
    )
)

# Compute ITR from run-level accuracy
run_summary["ITR"] = run_summary["accuracy"].apply(
    lambda acc: compute_itr(
        accuracy=acc,
        n_classes=N_CLASSES,
        trial_duration=TRIAL_DURATION
    )
)

# Sort rows for readability
run_summary = run_summary.sort_values(
    by=["condition", "run"]
).reset_index(drop=True)

print("run_summary shape:", run_summary.shape)
run_summary

# %% [markdown]
# ### Save outputs

# %%
# Save trial-level results
trial_results.to_csv(OUTPUT_DIR / "trial_results.csv", index=False)

# Save run-level summary
run_summary.to_csv(OUTPUT_DIR / "run_summary.csv", index=False)

print("Saved:")
print(OUTPUT_DIR / "trial_results.csv")
print(OUTPUT_DIR / "run_summary.csv")

# %%
