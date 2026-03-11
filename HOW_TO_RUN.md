# How to run the BCI keyboard

## Preview the keyboards (no EEG needed)

To see what the **calibration** vs **spelling** keyboard look like without connecting the Cyton:

1. Run: `python preview_keyboards.py` (or double-click a .bat that runs it).
2. First screen: **Calibration keyboard** (colored squares only, one highlighted). Press **SPACE** to continue.
3. Second screen: **Spelling keyboard** (squares + letters, "Your text: …"). Press **ESC** to quit.

---

## Quick run (Windows)

1. **Connect your OpenBCI Cyton** (dongle + board) and turn it on.
2. **Double‑click** one of these in the project folder:
   - `run_blueyellow.bat` — blue/yellow flicker version
   - `run_redgreen.bat` — red/green flicker version
3. The PsychoPy window will open. Press **Escape** to quit.

If you use the `pyenv` folder as your virtual environment, the batch file will activate it and then run the script.

---

## Run from terminal (any OS)

1. Open a terminal in the project folder:  
   `c:\Users\kimbr\Desktop\my_bci_project`
2. Activate the virtual environment (if you have one):
   - **Windows:** `pyenv\Scripts\activate`
   - **Mac/Linux:** `source pyenv/bin/activate`
3. Run the script:
   - Blue/yellow: `python run_vep_blueyellow.py`
   - Red/green: `python run_vep_redgreen.py`

---

## Before running

- **EEG hardware:** The script expects the OpenBCI Cyton to be connected. If the dongle isn’t found, you’ll get an error like “Cannot find OpenBCI port.”
- **Calibration vs spelling:** In the script, set `calibration_mode = True` (line 19) for calibration, or `False` for spelling.
- **Dependencies:** Need `psychopy`, `numpy`, `scipy`, `mne`, `brainflow`. If something is missing, install with:  
  `pip install psychopy numpy scipy mne brainflow`

---

## Will it run properly?

- **With Cyton connected and drivers OK:** Yes — calibration and spelling should run with the new calibration (squares only) vs spelling (letter keyboard) behavior.
- **Without the board:** The script will stop when it tries to find the OpenBCI port; it doesn’t have a “no hardware” mode.
