# What to upload to GitHub

## ✅ Upload these (your project code and docs)

| Item | Why |
|------|-----|
| **run_vep_blueyellow.py** | Main BCI run script (blue/yellow) |
| **run_vep_redgreen.py** | Main BCI run script (red/green) |
| **preview_keyboards.py** | Preview calibration vs spelling (no hardware) |
| **run_blueyellow.bat** | Quick run for blue/yellow (Windows) |
| **run_redgreen.bat** | Quick run for red/green (Windows) |
| **preview_keyboards.bat** | Quick run for preview (Windows) |
| **scripts/** | Training scripts (e.g. train_trca_blueyellow.py, train_trca_redgreen.py) |
| **HOW_TO_RUN.md** | How to run the project |
| **DATA_AND_SAVING.md** | How calibration data is saved |
| **.gitignore** | Tells Git what not to track |
| **requirements.txt** | Python dependencies so others can `pip install -r requirements.txt` |

## ❌ Do NOT upload these

| Item | Why |
|------|-----|
| **pyenv/** | Virtual environment. Others run `python -m venv pyenv` and `pip install -r requirements.txt`. |
| **data/** | Recorded EEG (large, personal). Others run calibration to get their own. |
| **cache/** | Trained models (can be re-trained). Or upload a small example if you want. |
| **1.24.1**, **5.1.2** | Look like version/build folders; not needed in the repo. |

## 🤔 Brainda

Your training scripts use **brainda** (e.g. `from brainda.algorithms...`). You have a **brainda** folder in the project.

- **Option A – Don’t upload brainda:** If brainda is on PyPI, add it to a `requirements.txt` and others run `pip install brainda`. Then add `brainda/` to `.gitignore`.
- **Option B – Upload brainda:** If you use a local/custom copy, keep the folder and do **not** add `brainda/` to `.gitignore` so it gets pushed.

Right now `.gitignore` does **not** ignore `brainda/`, so if you push the whole project, brainda will be included. If you prefer Option A, uncomment the `brainda/` line in `.gitignore`.

## Quick checklist

1. Create the repo on GitHub.
2. In your project folder: `git init`, then add and commit the files listed under “Upload these” (and `scripts/`, `.gitignore`). Don’t add `pyenv/`, `data/`, `cache/`.
3. Add a `requirements.txt` with the main dependencies (psychopy, numpy, scipy, mne, brainflow, brainda if from PyPI, etc.) so others can install them.
4. Push to GitHub.
