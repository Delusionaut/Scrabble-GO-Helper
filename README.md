# Scrabble GO Helper

Reads the live Scrabble GO board and rack from a tablet over ADB, then shows the highest-scoring move.

## Run

Double-click `run\ScrabbleGOHelper.exe`.

The window has **Connect** and **Capture & Solve**. The device box defaults to the tablet address used by the project.

## Build

`run\` is not committed. Rebuild the executable from `source\`:

```
cd source
pip install numpy opencv-python pyinstaller
pyinstaller --noconfirm --onedir --add-data "dict;dict" --add-data "templates.npz;." app.py
```

Then copy `dist\app` to `run\` and rename the exe to `ScrabbleGOHelper.exe`.

`source\` is the Python project the executable was built from.
