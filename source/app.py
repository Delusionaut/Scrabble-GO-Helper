"""Scrabble GO helper - desktop GUI.

Reads the live board + rack from a Samsung Galaxy Tab A9 over ADB, transcribes
it with the template-matching vision pipeline, solves for the highest-scoring
move using the NWL2023 dictionary, and draws EXACTLY where to place the tiles
onto the captured screenshot.

Every capture re-reads the WHOLE board and re-solves from scratch, so the app
always reflects whatever board is on screen right now (not a previous one).

Run:
    python app.py
"""
import base64
import datetime
import faulthandler
import os
import subprocess
import sys
import threading
import traceback

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import vision
import solver

def _base_dir():
    """Directory that holds bundled data (source tree, or the frozen bundle)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _runtime_dir():
    """Writable directory next to the script or the built executable."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


HERE = _base_dir()
RUNTIME = _runtime_dir()
TEMPLATES = os.path.join(HERE, "templates.npz")
# NWL2023 = the lexicon Scrabble GO uses for "English - US".
WORDFILE = os.path.join(HERE, "dict", "nwl2023.txt")
SIZE = 15
COLS = "ABCDEFGHIJKLMNO"

# Prefer the full platform-tools path so the GUI works regardless of PATH.
_ADB = r"C:\Users\Windows\AppData\Local\Android\Sdk\platform-tools\adb.exe"
ADB = _ADB if os.path.exists(_ADB) else "adb"

DEFAULT_SERIAL = "192.168.4.42:40237"

# --------------------------------------------------------------------------- #
# Crash logging
# --------------------------------------------------------------------------- #
# Previously an intermittent failure left NO trace: Windows logged no
# Application Error for python.exe and tkinter exits silently when an uncaught
# exception bubbles out of mainloop. Log everything to crash.log instead.
LOG_PATH = os.path.join(RUNTIME, "crash.log")
try:
    _LOGF = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
except OSError:
    _LOGF = None


def log(msg):
    """Append a timestamped line to crash.log. Never raises."""
    if _LOGF is None:
        return
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _LOGF.write("[%s] %s\n" % (ts, msg))
        _LOGF.flush()
    except Exception:
        pass


def install_crash_logging():
    """Capture hard crashes (segfaults) and uncaught exceptions to crash.log."""
    try:
        faulthandler.enable(_LOGF, all_threads=True)
    except Exception:
        pass

    def _excepthook(t, v, tb):
        log("UNCAUGHT EXCEPTION:\n"
            + "".join(traceback.format_exception(t, v, tb)))
        sys.__excepthook__(t, v, tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        name = args.thread.name if args.thread else "?"
        log("UNCAUGHT THREAD EXCEPTION in %s:\n%s"
            % (name, "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback))))

    threading.excepthook = _thread_excepthook
    log("=== app start (python %s) ===" % sys.version.split()[0])


# --------------------------------------------------------------------------- #
# ADB capture
# --------------------------------------------------------------------------- #
def adb_run(args, serial=None, binary=True, timeout=15):
    """Run an adb command, always bounded by a timeout.

    The tablet's wireless-debugging TCP port changes every time it reconnects
    (adb.log shows minutes of failed reconnect attempts to a stale port). A
    blocking adb call with no timeout on the GUI thread was the main cause of
    the app appearing to "crash" (it actually froze).
    """
    cmd = [ADB]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        log("adb timeout (%ss): %s" % (timeout, " ".join(args)))
        if binary:
            return b""
        return subprocess.CompletedProcess(cmd, 1, b"", b"adb timeout")
    except OSError as e:
        log("adb spawn failed: %s" % e)
        if binary:
            return b""
        return subprocess.CompletedProcess(cmd, 1, b"", str(e).encode())
    return p.stdout if binary else p


def adb_capture(serial=None):
    """Return a BGR screenshot from the device, robust to Windows CRLF issues."""
    cmd = [ADB]
    if serial:
        cmd += ["-s", serial]
    cmd += ["exec-out", "screencap", "-p"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=20)
    except subprocess.TimeoutExpired:
        log("adb screencap timed out after 20s")
        raise RuntimeError("adb screencap timed out after 20s - device offline?")
    raw = proc.stdout
    if not raw:
        err = proc.stderr.decode("utf-8", "ignore").strip()
        raise RuntimeError("adb returned no data. " + (err or "Is the device connected?"))

    def _decode(buf):
        arr = np.frombuffer(buf, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    img = _decode(raw)
    if img is None:
        # legacy adb translated \n -> \r\n inside the stream
        img = _decode(raw.replace(b"\r\n", b"\n"))
    if img is None:
        # last resort: slice from the PNG signature to the IEND chunk
        s = raw.find(b"\x89PNG")
        e = raw.rfind(b"IEND")
        if s != -1 and e != -1:
            img = _decode(raw[s:e + 8])
    if img is None:
        raise RuntimeError("failed to decode screenshot (%d bytes)" % len(raw))
    return img


def adb_connect(serial):
    out = adb_run(["connect", serial], binary=False)
    return (out.stdout + out.stderr).decode("utf-8", "ignore").strip()


def list_devices():
    """Return [(serial, state), ...] from `adb devices`."""
    out = adb_run(["devices"], binary=False)
    text = (out.stdout + out.stderr).decode("utf-8", "ignore")
    devs = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            devs.append((parts[0], parts[1]))
    return devs


def resolve_serial(preferred):
    """Return a usable serial: the preferred one if online, else the only device.

    The tablet's ADB port changes every time wireless debugging reconnects, so
    an exact match is not required when exactly one device is attached.
    """
    devs = [s for (s, st) in list_devices() if st == "device"]
    if preferred and preferred in devs:
        return preferred
    if not devs:
        return preferred or None
    if len(devs) == 1:
        return devs[0]
    # several devices: prefer a plain TCP/IP target over an mDNS adb-* name
    tcp = [s for s in devs if ":" in s and not s.startswith("adb-")]
    return tcp[0] if tcp else devs[0]


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #
def _cell_rect(info, r, c):
    x0, y0, x1, y1 = info["bbox"]
    cw = (x1 - x0) / SIZE
    ch = (y1 - y0) / SIZE
    return int(x0 + c * cw), int(y0 + r * ch), int(cw), int(ch)


def make_overlay(img, lines, info, rack, move=None):
    """Annotate the capture. If `move` is given, mark exactly where to place it.

    move = (score, word, placed, start_cell, direction) as returned by solver.
    placed = [(row, col, LETTER), ...] for the NEW tiles only.
    """
    vis = img.copy()
    x0, y0, x1, y1 = info["bbox"]
    cw = (x1 - x0) / SIZE
    ch = (y1 - y0) / SIZE

    # recognised tiles + a faint grid
    for rr in range(SIZE):
        for cc in range(SIZE):
            px, py, w, h = _cell_rect(info, rr, cc)
            cv2.rectangle(vis, (px, py), (px + w, py + h), (90, 90, 90), 1)
            L = lines[rr][cc]
            if L != ".":
                color = (0, 0, 255) if L == "?" else (0, 170, 0)
                cv2.rectangle(vis, (px, py), (px + w, py + h), color, 2)
                cv2.putText(vis, L, (px + 5, py + h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # coordinate labels drawn on the image so nothing has to be counted by eye
    for cc in range(SIZE):
        px, py, w, h = _cell_rect(info, 0, cc)
        cv2.putText(vis, COLS[cc], (px + w // 3, max(20, py - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for rr in range(SIZE):
        px, py, w, h = _cell_rect(info, rr, 0)
        cv2.putText(vis, str(rr + 1), (max(5, px - int(cw * 0.75)), py + int(h * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if move is None:
        cv2.putText(vis, "RACK: " + rack, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return vis

    score, full, placed, start_cell, direction = move
    if not placed:
        return vis

    # translucent wash over the destination squares
    wash = vis.copy()
    for (r, c, L) in placed:
        px, py, w, h = _cell_rect(info, r, c)
        cv2.rectangle(wash, (px, py), (px + w, py + h), (0, 220, 255), -1)
    cv2.addWeighted(wash, 0.45, vis, 0.55, 0, vis)

    # highlight each destination square, with the letter and placement order
    for i, (r, c, L) in enumerate(placed, 1):
        px, py, w, h = _cell_rect(info, r, c)
        cv2.rectangle(vis, (px, py), (px + w, py + h), (0, 0, 255), 5)
        cv2.putText(vis, L.upper(), (px + int(w * 0.18), py + int(h * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 5)
        cv2.putText(vis, str(i), (px + 4, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

    # arrow showing the direction of play
    r0, c0, _ = placed[0]
    r1, c1, _ = placed[-1]
    ax0, ay0, aw, ah = _cell_rect(info, r0, c0)
    ax1, ay1, _, _ = _cell_rect(info, r1, c1)
    if (r0, c0) != (r1, c1):
        cv2.arrowedLine(vis, (ax0 + aw // 2, ay0 + ah // 2),
                        (ax1 + aw // 2, ay1 + ah // 2),
                        (0, 0, 255), 6, tipLength=0.25)

    # banner
    hdr = vis.shape[1]
    cv2.rectangle(vis, (0, 0), (hdr, 70), (0, 0, 0), -1)
    cv2.putText(vis, "PLAY '%s' %s  -  %d pts" %
                (full, "DOWN" if direction == "V" else "ACROSS", score),
                (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)
    cv2.putText(vis, "Place tiles 1->%d in the red squares. "
                     "Green letters are already on the board - do not move them."
                % len(placed),
                (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return vis


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scrabble GO Helper")
        self.geometry("1280x860")
        self.templates = vision.load_templates(TEMPLATES)
        self.last_img = None
        self.last_info = None
        self._photo = None
        self._res = []
        self._lines = ["." * 15] * 15
        self._sig = None          # signature of the last read board/rack
        self._busy = False        # a solve is currently in flight
        self._result_box = None   # (res, dt) filled by the solver thread
        self._solve_err = None    # exception filled by the solver thread
        self._build()
        if not self.templates:
            self.set_status("WARNING: templates.npz not found - run build_templates.py")
        if not os.path.exists(WORDFILE):
            self.set_status("WARNING: %s not found - solver will fail" % WORDFILE)

    # -- layout --------------------------------------------------------------
    def _build(self):
        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Device:").pack(side="left")
        self.serial = tk.StringVar(value=DEFAULT_SERIAL)
        ttk.Entry(top, textvariable=self.serial, width=22).pack(side="left", padx=4)
        ttk.Button(top, text="Connect", command=self.on_connect).pack(side="left")
        ttk.Button(top, text="Capture & Solve", command=self.on_capture).pack(side="left", padx=4)

        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)

        # left: the annotated "where to place" image
        left = ttk.LabelFrame(body, text="Where to place the word", padding=6)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Label(left, background="#222")
        self.canvas.pack(fill="both", expand=True)

        # right: board, rack, move list
        right = ttk.Frame(body)
        right.pack(side="left", fill="y", padx=8)

        bf = ttk.LabelFrame(right, text="Board ('.'=empty, edit if wrong)", padding=6)
        bf.pack(fill="x")
        self.board_text = tk.Text(bf, width=20, height=15, font=("Consolas", 13))
        self.board_text.pack()
        self.board_text.insert("1.0", "\n".join(["." * 15] * 15))

        rk = ttk.LabelFrame(right, text="Rack ('?' = blank)", padding=6)
        rk.pack(fill="x", pady=6)
        self.rack = tk.StringVar()
        ttk.Entry(rk, textvariable=self.rack, width=16,
                  font=("Consolas", 15)).pack(side="left")
        ttk.Button(rk, text="Find Best Word",
                   command=self.on_solve).pack(side="left", padx=6)

        res = ttk.LabelFrame(right, text="Moves (double-click = show on image)", padding=6)
        res.pack(fill="both", expand=True, pady=6)
        cols = ("score", "word", "dir", "at")
        self.tree = ttk.Treeview(res, columns=cols, show="headings", height=16)
        for c, w in zip(cols, (60, 110, 45, 60)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self.on_pick_move)

        self.status = tk.StringVar(value="Ready. Press 'Capture & Solve'.")
        ttk.Label(self, textvariable=self.status, relief="sunken",
                  anchor="w").pack(fill="x", side="bottom")

    def set_status(self, msg):
        self.status.set(msg)

    # -- image display -------------------------------------------------------
    def _show(self, bgr):
        h, w = bgr.shape[:2]
        maxw, maxh = 820, 820
        s = min(maxw / w, maxh / h, 1.0)
        small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", small)
        self._photo = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
        self.canvas.configure(image=self._photo)

    # -- actions -------------------------------------------------------------
    def on_connect(self):
        try:
            msg = adb_connect(self.serial.get().strip())
            serial = resolve_serial(self.serial.get().strip() or None)
            if serial:
                self.serial.set(serial)
            devs = ", ".join("%s (%s)" % d for d in list_devices()) or "no devices"
            self.set_status("%s | devices: %s" % (msg, devs))
        except Exception as e:
            messagebox.showerror("ADB", str(e))

    def on_capture(self):
        try:
            self._capture(solve=True)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Capture failed", str(e))
            self.set_status("Capture failed: %s" % e)

    def _capture(self, solve=True, quiet=False):
        if not quiet:
            self.set_status("Capturing...")
            self.update_idletasks()
        serial = resolve_serial(self.serial.get().strip() or None)
        if serial:
            self.serial.set(serial)
        img = adb_capture(serial)
        return self._read(img, solve=solve,
                          note="Captured %dx%d from %s" %
                               (img.shape[1], img.shape[0], serial or "default device"))

    def _read(self, img, solve=False, note=None):
        """Read the FULL board + rack from `img` and reset all dependent state."""
        self.last_img = img
        lines, rack, info = vision.read_game(img, self.templates)
        self.last_info = info
        self._lines = lines
        self._res = []
        self._sig = (tuple(lines), rack)
        # reset the board text, rack, move list and overlay for THIS capture
        self.board_text.delete("1.0", "end")
        self.board_text.insert("1.0", "\n".join(lines))
        self.rack.set(rack)
        for i in self.tree.get_children():
            self.tree.delete(i)
        self._show(make_overlay(img, lines, info, rack))
        n = sum(1 for r in lines for ch in r if ch != ".")
        msg = "Read %d tiles, rack=%s, %d uncertain." % (n, rack, len(info["unknown"]))
        if note:
            msg = note + " | " + msg
        self.set_status(msg)
        if solve:
            self.on_solve()

    def _board_lines(self):
        raw = self.board_text.get("1.0", "end").splitlines()
        out = []
        for i in range(SIZE):
            row = (raw[i] if i < len(raw) else "").strip().upper()
            row = row.replace(" ", "").replace("-", "")
            out.append((row + "." * SIZE)[:SIZE])
        return out

    def on_solve(self):
        if self._busy:
            return
        rack = self.rack.get().strip().upper() or "?"
        self._lines = self._board_lines()
        self._busy = True
        self._result_box = None
        self._solve_err = None
        self.set_status("Solving...")
        self.update_idletasks()

        grid = [row[:] for row in self._lines]

        def work():
            # Runs off the main thread; DO NOT touch Tk here. Just hand the
            # result to the main-thread poller, which is the only safe way to
            # update widgets (calling self.after() from this thread raises
            # 'main thread is not in main loop' and silently froze the app).
            try:
                self._result_box = solver.solve(grid, rack, WORDFILE)
            except Exception as e:
                self._solve_err = e
                log("solver error: %s" % e)

        threading.Thread(target=work, daemon=True).start()
        self._poll_solver()

    def _poll_solver(self):
        """Main-thread poller: applies the solver result when it is ready."""
        if self._result_box is not None:
            res, dt = self._result_box
            self._result_box = None
            self._busy = False
            self._show_results(res, dt)
            return
        if self._solve_err is not None:
            err = self._solve_err
            self._solve_err = None
            self._busy = False
            traceback.print_exception(type(err), err, err.__traceback__)
            self.set_status("Solver failed: %s" % err)
            return
        self.after(50, self._poll_solver)

    def _show_results(self, res, dt):
        self._busy = False
        self._res = res
        for i in self.tree.get_children():
            self.tree.delete(i)
        for sc, full, placed, start_cell, d in res[:80]:
            at = "%s%d" % (COLS[start_cell[1]], start_cell[0] + 1)
            self.tree.insert("", "end", values=(sc, full, d, at))
        if res:
            self._annotate(res[0])
            self.set_status("Found %d moves in %.2fs. Best: %s for %d pts (%s)."
                            % (len(res), dt, res[0][1], res[0][0],
                               "down" if res[0][4] == "V" else "across"))
        else:
            self.set_status("Found 0 moves in %.2fs - check the board/rack text." % dt)

    def _annotate(self, move):
        if self.last_img is None or self.last_info is None:
            return
        vis = make_overlay(self.last_img, self._lines, self.last_info,
                           self.rack.get(), move)
        self._show(vis)

    def on_pick_move(self, _evt):
        sel = self.tree.selection()
        if not sel or not self._res:
            return
        idx = self.tree.index(sel[0])
        if 0 <= idx < len(self._res):
            self._annotate(self._res[idx])
            self.set_status("Showing move: %s" % self._res[idx][1])

if __name__ == "__main__":
    install_crash_logging()
    App().mainloop()
