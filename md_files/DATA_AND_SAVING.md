# How the collected data gets saved (step-by-step)

This applies when you run with **calibration_mode = True** and the **OpenBCI Cyton** connected. Only calibration runs save data to disk; spelling mode uses the model in memory and does not write new EEG files.

---

## 1. Where files are saved

At the top of `run_vep_blueyellow.py` (or `run_vep_redgreen.py`) the save directory is set from:

- **subject** (e.g. `1`)
- **session** (e.g. `1`)
- **stim_type** (e.g. `'alternating'`)
- **stim_duration** (e.g. `1.2`)
- **run** (e.g. `1`)

Example:

- **save_dir** = `data/blueyellow_cyton8_alternating-vep_32-class_1.2s/sub-01/ses-01/`
- **save_file_eeg** = `.../eeg_2-per-class_run-1.npy`
- **save_file_aux** = `.../aux_2-per-class_run-1.npy`
- **save_file_eeg_trials** = `.../eeg-trials_2-per-class_run-1.npy`
- **save_file_aux_trials** = `.../aux-trials_2-per-class_run-1.npy`

So: **all saved data goes under `data/` in the project folder**, in the path above.

---

## 2. When you start calibration

- The script connects to the Cyton and starts streaming.
- It creates empty arrays: **eeg**, **aux**, **timestamp**, and lists **eeg_trials**, **aux_trials**.
- A background thread continuously reads from the board and puts chunks **(eeg_in, aux_in, timestamp_in)** into a **queue** (in memory). Nothing is written to disk yet.

---

## 3. During each calibration trial

For every trial the script:

1. **Shows the calibration keyboard** (squares only, one highlighted).
2. **Starts the flicker** for `stim_duration` seconds (e.g. 1.2 s). The photosensor (if used) goes on during flicker.
3. **After the flicker ends**, it waits until enough data has been collected from the queue for this trial.
4. **Finds trial boundaries** using the **aux** channel (photosensor):  
   - **trial_starts** = where the signal goes high (flicker on).  
   - **trial_ends** = where it goes low (flicker off).
5. **Cuts one segment** of EEG and aux:
   - From **trial_start** (with a small baseline before) to **trial_start + trial_duration**.
   - **trial_eeg** = filtered (e.g. 2–40 Hz) and baseline-corrected.
   - **trial_aux** = same time segment from the aux channels.
6. **Appends** these to lists in memory:
   - **eeg_trials.append(trial_eeg)**
   - **aux_trials.append(trial_aux)**
7. The **continuous** **eeg** and **aux** arrays are also extended with all new data from the queue (for the whole run).

So during the run, **everything stays in memory** (queue + **eeg** / **aux** / **eeg_trials** / **aux_trials**). No saving yet.

---

## 4. When data is actually written to disk

Saving happens in two cases:

### A. Calibration finishes normally

After the last calibration trial, the script does:

1. **Create the folder** (if needed):  
   `os.makedirs(save_dir, exist_ok=True)`
2. **Save continuous data:**
   - **np.save(save_file_eeg, eeg)** → all EEG channels for the whole run.
   - **np.save(save_file_aux, aux)** → all aux channels for the whole run.
3. **Save trial data:**
   - **np.save(save_file_eeg_trials, eeg_trials)** → list of per-trial EEG segments (filtered, baseline-corrected).
   - **np.save(save_file_aux_trials, aux_trials)** → list of per-trial aux segments.
4. **Stop the board:**  
   `board.stop_stream()` and `board.release_session()`.

So after a full calibration run, you get **four .npy files** in **save_dir**.

### B. You press Escape during calibration

If you press **Escape** in the middle of calibration:

1. The script still **creates save_dir** and **saves the same four arrays** (eeg, aux, eeg_trials, aux_trials) with whatever data was collected so far.
2. Then it stops the board and quits.

So even when you quit early, **everything collected up to that point is saved** in the same way.

---

## 5. What each saved file contains

| File | Content |
|------|--------|
| **eeg_2-per-class_run-1.npy** | Continuous EEG (all channels) for the whole run. Shape like (8, n_samples) for 8 channels. |
| **aux_2-per-class_run-1.npy** | Continuous aux (e.g. photosensor) for the whole run. Shape like (3, n_samples). |
| **eeg-trials_2-per-class_run-1.npy** | Python list of arrays; each array is one trial’s EEG (channels × time), filtered and baseline-corrected. |
| **aux-trials_2-per-class_run-1.npy** | Python list of arrays; each array is one trial’s aux for the same time window. |

The **trial** files are what you normally use for training (e.g. TRCA/FBTRCA): each trial has a known target index from **trial_sequence**, so you have (EEG segment, target_id) pairs.

---

## 6. Spelling mode (calibration_mode = False)

In spelling mode:

- EEG is still read and **eeg_trials** / **aux_trials** are still appended in memory for each “trial” (each selection).
- The script **does not** call **np.save** for these; it only uses the last trial to run the model and get a **prediction**.
- So **no new EEG or aux files are written** during spelling. Only calibration runs produce the saved data described above.

---

## 7. Quick summary

1. **Save location:** `data/blueyellow_cyton8_.../sub-XX/ses-XX/` (or redgreen equivalent).
2. **When:** Only when **calibration_mode = True**, either when all trials finish or when you press Escape.
3. **What:** Four .npy files: continuous **eeg** and **aux**, and per-trial **eeg_trials** and **aux_trials**.
4. **Spelling:** Uses the model only; does not save new EEG/aux files.
