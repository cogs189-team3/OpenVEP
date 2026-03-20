"""
Preview the calibration keyboard vs the spelling keyboard (no EEG hardware needed).
Run this script and you'll see each screen for a few seconds. Press SPACE to advance, ESC to quit.
"""
from psychopy import visual, core
import numpy as np
import psychopy.event

# Same layout as run_vep_blueyellow.py (blue/yellow)
width = 1536
height = 864
aspect_ratio = width / height

letters = 'QAZ⤒WSX,EDC?R⌫FVT⎵GBYHN.UJMPIKOL'

def make_column_colors():
    colors = []
    for i_col in range(8):
        col_color = [-1, -1, 1] if i_col % 2 == 0 else [1, 1, -1]
        for _ in range(4):
            colors.append(col_color)
    return np.array(colors)

key_bg_colors = make_column_colors()

# Preview window (smaller, not fullscreen)
window = visual.Window(
    size=[width, height],
    fullscr=False,
    allowGUI=True,
)

def create_32_target_positions(size=2/8*0.7):
    size_with_border = size / 0.7
    w, h = window.size
    ar = w / h
    positions = []
    for i_col in range(8):
        for j_row in range(4):
            x = i_col*size_with_border - 1 + size_with_border/2
            y = -j_row*size_with_border*ar + 1 - size_with_border*ar/2 - 1/4/2
            positions.append((float(x), float(y)))
    return positions

def create_32_squares():
    """Draw 32 colored squares as individual ShapeStims (avoids ElementArrayStim array bugs in PsychoPy)."""
    size = 2/8*0.7
    w, h = window.size
    ar = w / h
    half_w = size / 2
    half_h = (size * ar) / 2
    positions = create_32_target_positions(size)
    squares = []
    for i, pos in enumerate(positions):
        cx, cy = pos
        verts = [(cx - half_w, cy - half_h), (cx + half_w, cy - half_h),
                 (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)]
        rgb = key_bg_colors[i].tolist() if hasattr(key_bg_colors[i], 'tolist') else list(key_bg_colors[i])
        sq = visual.ShapeStim(win=window, vertices=verts, units='norm', lineColor=None, fillColor=rgb, closeShape=True)
        squares.append(sq)
    return squares

def create_letter_labels():
    """Draw letters as TextStims (avoids numpy array texture/mask issues in PsychoPy)."""
    positions = create_32_target_positions(size=2/8*0.7)
    stims = []
    for i, (pos, letter) in enumerate(zip(positions, letters)):
        s = visual.TextStim(window, text=letter, pos=pos, color='white', units='norm', height=0.08, alignText='center')
        stims.append(s)
    return stims

target_positions = create_32_target_positions(size=2/8*0.7)
square_stims = create_32_squares()
letter_labels = create_letter_labels()

# ─── 1) CALIBRATION KEYBOARD (squares only, no letters) ───
calib_title = visual.TextStim(
    window, text='CALIBRATION — Look at the highlighted square',
    pos=(0, 1-0.05), color='white', units='norm', height=0.06
)
trial_text = visual.TextStim(
    window, text='Trial 1/64 (preview)',
    pos=(0, -1+0.07), color='white', units='norm', height=0.07
)
def make_highlight_rect(center_pos, size_scale=1.3):
    """Outline rectangle as ShapeStim (avoids Rect vertices bug in PsychoPy)."""
    size = 2/8*0.7 * size_scale
    half_w = size / 2
    half_h = (size * aspect_ratio) / 2
    cx, cy = center_pos[0], center_pos[1]
    verts = [(cx - half_w, cy - half_h), (cx + half_w, cy - half_h),
             (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)]
    return visual.ShapeStim(win=window, vertices=verts, units='norm', lineColor='white', lineWidth=3, fillColor=None, closeShape=True)

aim_target = make_highlight_rect(target_positions[0])

for i, sq in enumerate(square_stims):
    sq.fillColor = [-key_bg_colors[i][0], -key_bg_colors[i][1], -key_bg_colors[i][2]]
    sq.draw()
calib_title.draw()
trial_text.draw()
aim_target.draw()
window.flip()

msg = visual.TextStim(window, text='CALIBRATION KEYBOARD (squares only)\nTake a moment to look. Press SPACE when ready for spelling keyboard — ESC to quit', pos=(0, -0.85), color='white', units='norm', height=0.05)
msg.draw()
window.flip()

# Give the screen time to stay visible before we start checking keys
core.wait(1.5)

waiting = True
while waiting:
    keys = psychopy.event.getKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        window.close()
        core.quit()
    if 'space' in keys:
        waiting = False
    core.wait(0.05)

# ─── 2) SPELLING KEYBOARD (squares + letters) ───
spelling_title = visual.TextStim(
    window, text='SPELLING — Look at the key you want to type',
    pos=(0, 1-0.05), color='white', units='norm', height=0.06
)
pred_text = visual.TextStim(
    window, text='Your text: hello (preview)',
    pos=(0.07, 1-0.14), color='white', units='norm', height=0.08, alignText='left', wrapWidth=1.94
)
pred_target = make_highlight_rect(target_positions[5])

for i, sq in enumerate(square_stims):
    sq.fillColor = [-key_bg_colors[i][0], -key_bg_colors[i][1], -key_bg_colors[i][2]]
    sq.draw()
for lab in letter_labels:
    lab.draw()
spelling_title.draw()
pred_text.draw()
pred_target.draw()
msg2 = visual.TextStim(window, text='SPELLING KEYBOARD (with letters)\nPress ESC to quit', pos=(0, -0.85), color='white', units='norm', height=0.05)
msg2.draw()
window.flip()

while True:
    keys = psychopy.event.getKeys(keyList=['escape'])
    if 'escape' in keys:
        break
    core.wait(0.05)

window.close()
