# Scrabble GO Helper

Desktop app that reads the live Scrabble GO board and rack from a Samsung
Galaxy Tab A9 over ADB, transcribes it, and finds the highest-scoring move on
demand via a button press.

## Requirements
- Python 3.10+ with `numpy`, `opencv-python`, `tkinter` (all present here).
- Android platform-tools (`adb`) on PATH.
- Tablet with USB/network debugging enabled, matched to the same screen the app
  is played on. Current device: `192.168.4.42:35189`.

## Run
```
cd C:/Users/Windows/scrabble_proj
python app.py
```

## Workflow
1. **Connect** — runs `adb connect <device>` for the address in the Device box.
2. **Capture & Read** — takes a live screenshot (`adb exec-out screencap -p`),
   detects the 15x15 board, reads every occupied tile and the rack using the
   template library (`templates.npz`), and fills the board text box + rack.
3. Edit any cell by hand if needed (`?` marks a low-confidence tile).
4. **Find Best Word** — runs the dictionary solver (`dict/csw.txt`, Collins
   Scrabble Words) and lists the top moves with score, direction, start square,
   and the exact tiles to place.
5. **Save Debug Overlay** — writes `debug_overlay.png` showing every recognized
   tile boxed in green (red = uncertain) plus the parsed rack.

## Files
- `app.py` — the GUI (capture, read, solve, overlay).
- `vision.py` — board detection, tile/glyph extraction, letter classification.
- `solver.py` — move generator + scorer (trie-based, blanks supported).
- `build_templates.py` — regenerates `templates.npz` from a screenshot.
- `templates.npz` — 26 letter glyph templates (A–Z).

## Confidence / tuning
- `vision.UNKNOWN_DIST` (default 4200) and `vision.UNKNOWN_MARGIN` (default 60)
  control when a glyph is left as `?` instead of a guessed letter.
- Blank tiles can be entered as `?` in the rack field.