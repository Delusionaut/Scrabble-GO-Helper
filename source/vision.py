import os
import numpy as np
import cv2

SIZE = 15
GLYPH = 32  # normalized glyph size


def _chan(img):
    return img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)


def _comps(mask, H, min_area=1800, max_area=6500, minwh=45, maxwh=90,
           ymin_frac=0.20):
    """Centroids of connected premium-square blobs of a plausible cell size."""
    m = mask.astype(np.uint8)
    m[:int(H * ymin_frac), :] = 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if min_area < a < max_area and minwh <= w <= maxwh and minwh <= h <= maxwh:
            out.append((float(cent[i][0]), float(cent[i][1])))
    return out


def _best_spacing(vals, lo=54.0, hi=92.0, step=0.25):
    """Estimate the cell pitch of an evenly spaced grid from centroid coords.

    Uses a circular-mean (phase) fit: for the true pitch every centroid shares
    the same phase, so the mean resultant length R peaks at the correct pitch.
    Returns (pitch, R, phase) with phase = (origin + pitch/2) mod pitch.
    """
    vals = np.asarray(vals, float)
    best = (None, -1.0, 0.0)
    for cw in np.arange(lo, hi + 1e-9, step):
        z = np.exp(2j * np.pi * vals / cw)
        R = abs(z.mean())
        if R > best[1]:
            ph = (np.angle(z.mean()) % (2 * np.pi)) / (2 * np.pi) * cw
            best = (cw, R, ph)
    return best


# Canonical premium-square layout (grid indices) for a standard Scrabble board.
TW_CELLS = {(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)}
DW_CELLS = {(1, 1), (1, 13), (2, 2), (2, 12), (3, 3), (3, 11), (4, 4), (4, 10),
            (10, 4), (10, 10), (11, 3), (11, 11), (12, 2), (12, 12),
            (13, 1), (13, 13), (7, 7)}
TL_CELLS = {(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13), (9, 1), (9, 5),
            (9, 9), (9, 13), (13, 5), (13, 9)}
DL_CELLS = {(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2),
            (6, 6), (6, 8), (6, 12), (7, 3), (7, 11), (8, 2), (8, 6), (8, 8),
            (8, 12), (11, 0), (11, 7), (11, 14), (12, 6), (12, 8),
            (14, 3), (14, 11)}
CANON = {'TW': TW_CELLS, 'DW': DW_CELLS, 'TL': TL_CELLS, 'DL': DL_CELLS}


def _align_origin(pts, cwx, phx, cwy, phy, W, H):
    """Resolve the whole-cell origin by aligning detected premium squares.

    Each premium square has a known (row, col); the correct origin makes every
    detected centroid land on a canonical premium cell of its own colour. This
    pins the board even when the outer TW squares are hidden by tiles."""
    bx = phx - cwx / 2.0
    by = phy - cwy / 2.0
    best = (-1, bx, by)
    for mx in range(-2, 20):
        x0 = bx + mx * cwx
        if x0 < -cwx or x0 > W:
            continue
        for my in range(-2, 20):
            y0 = by + my * cwy
            if y0 < -cwy or y0 > H:
                continue
            score = 0
            for (px, py, t) in pts:
                c = int(round((px - x0) / cwx - 0.5))
                r = int(round((py - y0) / cwy - 0.5))
                if (r, c) in CANON[t]:
                    score += 1
            if score > best[0]:
                best = (score, x0, y0)
    return best[1], best[2]


def detect_board(img):
    """Locate the 15x15 board from its premium squares.

    Primary path: the inset TW (magenta) squares at grid lines 0/7/14 give a
    precise cell size and origin. If a fully-tiled board hides the extreme TW
    squares, fall back to fitting a regular 15x15 grid to every visible premium
    square (TW/DW/TL/DL), which stays correct for any board state.
    """
    b, g, r = _chan(img)
    H, W = img.shape[:2]

    tw = (r > 200) & (g < 150) & (b > 100) & (b < 220)
    comps = _comps(tw, H)
    if len(comps) >= 4:
        xs = sorted(c[0] for c in comps)
        ys = sorted(c[1] for c in comps)
        cw = (xs[-1] - xs[0]) / 14.0
        ch = (ys[-1] - ys[0]) / 14.0
        if 54 <= cw <= 92 and 54 <= ch <= 92:
            return (xs[0] - cw / 2, ys[0] - ch / 2,
                    xs[-1] + cw / 2, ys[-1] + ch / 2)

    # Fallback: fit a regular grid to all visible premium squares.
    masks = [tw,
             (r > 225) & (g > 110) & (g < 200) & (b > 95) & (b < 185),   # DW
             (r < 95) & (g > 95) & (g < 170) & (b > 145),               # TL
             (r < 200) & (g > 180) & (b > 225)]                         # DL
    mask_types = [('TW', tw),
                  ('DW', (r > 225) & (g > 110) & (g < 200) & (b > 95) & (b < 185)),
                  ('TL', (r < 95) & (g > 95) & (g < 170) & (b > 145)),
                  ('DL', (r < 200) & (g > 180) & (b > 225))]
    pts = []
    for name, m in mask_types:
        for c in _comps(m, H):
            pts.append((c[0], c[1], name))
    uniq = []
    for p in pts:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 25 ** 2 for q in uniq):
            uniq.append(p)
    if len(uniq) < 4:
        raise RuntimeError('board not found (no premium squares detected)')
    xs = [p[0] for p in uniq]
    ys = [p[1] for p in uniq]
    cwx, rx, phx = _best_spacing(xs)
    cwy, ry, phy = _best_spacing(ys)
    if cwx is None or cwy is None or rx < 0.30 or ry < 0.30:
        raise RuntimeError('board not found (grid fit failed)')
    x0, y0 = _align_origin(uniq, cwx, phx, cwy, phy, W, H)
    return (x0, y0, x0 + 15 * cwx, y0 + 15 * cwy)


def _tile_face_mask(cell):
    """Boolean mask of tile-face pixels.

    Measured signatures:
      bright tile ~ (190,175,140)  g-b=35  r-b=50
      dim tile    ~ (165,160,133)  g-b=27  r-b=31
      yellow(move)~ (182,157, 47)  g-b=110 r-b=135
      empty       ~ (232,234,241)  g-b= -7 r-b= -9
      TW pink     ~ (239, 56,137)  g-b=-82 r-b=102
      DW red      ~ (255,156,134)  g-b= 22 r-b=121
      DL blue     ~ (155,209,255)  g-b=-46 r-b=-100
      TL blue     ~ ( 35,129,183)  g-b=-54 r-b=-148
    """
    b, g, r = _chan(cell)
    beige = (g - b >= 14) & (r - b >= 16) & (r - b <= 95)
    yellow = (g - b >= 55) & (r - b >= 40)
    return beige | yellow


def cell_occupied(img, x0, y0, x1, y1):
    """A cell is occupied if a large share of its interior is tile-face coloured."""
    ix = int((x1 - x0) * 0.16)
    iy = int((y1 - y0) * 0.16)
    cell = img[y0 + iy:y1 - iy, x0 + ix:x1 - ix]
    if cell.size == 0:
        return False, 0.0
    frac = _tile_face_mask(cell).mean()
    return frac > 0.25, frac


def _extract_glyph(img, x0, y0, x1, y1):
    """Crop the main letter from a cell (ignore the small subscript digit)."""
    cell = img[max(0, y0):y1, max(0, x0):x1]
    if cell.size == 0:
        return None
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bc, gc, rc = _chan(cell)
    # black letter strokes OR blue (opponent) letter strokes
    ink = ((gray < 115) | ((bc - rc) > 25)).astype(np.uint8) * 255
    th = ink
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    ch_cell = cell.shape[0]
    best = None
    best_score = 0
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # main letter is tall and reasonably large; subscript sits at the bottom
        if bh < ch_cell * 0.45 or area < 40:
            continue
        score = area
        if score > best_score:
            best_score = score
            best = (bx, by, bw, bh)
    if best is None:
        return None
    bx, by, bw, bh = best
    gimg = th[by:by + bh, bx:bx + bw]
    # aspect-preserving fit into GLYPH x GLYPH, centred (keeps 'I' thin)
    scale = float(GLYPH - 4) / max(bw, bh)
    nw = max(1, int(round(bw * scale)))
    nh = max(1, int(round(bh * scale)))
    gimg = cv2.resize(gimg, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((GLYPH, GLYPH), np.uint8)
    ox = (GLYPH - nw) // 2
    oy = (GLYPH - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = gimg
    return canvas


def read_board(img, bbox):
    x0, y0, x1, y1 = bbox
    cw = (x1 - x0) / SIZE
    ch = (y1 - y0) / SIZE
    grid = []
    glyphs = {}
    for rr in range(SIZE):
        row = []
        for cc in range(SIZE):
            cx0 = x0 + cc * cw
            cy0 = y0 + rr * ch
            occ, frac = cell_occupied(img, int(cx0), int(cy0), int(cx0 + cw), int(cy0 + ch))
            if occ:
                g = _extract_glyph(img, int(cx0), int(cy0), int(cx0 + cw), int(cy0 + ch))
                glyphs[(rr, cc)] = g
                row.append('?')
            else:
                row.append('.')
        grid.append(row)
    return grid, glyphs


def read_rack(img, board_bottom=None):
    """Find the player's rack tiles (bottom band) and return glyphs left->right."""
    H, W = img.shape[:2]
    mask = _tile_face_mask(img)
    ystart = int(board_bottom) + 8 if board_bottom else int(H * 0.78)
    band = mask[ystart:, :]
    rowcount = band.sum(axis=1)
    if rowcount.max() < 50:
        return []
    ys = np.where(rowcount > rowcount.max() * 0.2)[0]
    top = ystart + int(ys.min())
    bot = ystart + int(ys.max())
    strip = mask[top:bot + 1, :]
    colsum = strip.sum(axis=0)
    thr = max(10, strip.shape[0] * 0.25)
    spans = []
    c = 0
    while c < W:
        if colsum[c] > thr:
            s = c
            while c < W and colsum[c] > thr:
                c += 1
            if c - s > 40:
                spans.append((s, c))
        else:
            c += 1
    out = []
    for (s, e) in spans:
        sub = strip[:, s:e]
        rows = np.where(sub.any(axis=1))[0]
        y0, y1 = (top + int(rows.min()), top + int(rows.max())) if len(rows) else (top, bot)
        gph = _extract_glyph(img, s, y0, e, y1)
        out.append(gph)
    return out


def load_templates(path='templates.npz'):
    if not os.path.exists(path):
        return {}
    d = np.load(path)
    return {k: d[k] for k in d.files}


def save_templates(tpl, path='templates.npz'):
    np.savez(path, **tpl)


def classify(glyph, tpl):
    if glyph is None or not tpl:
        return '?'
    best, bd = '?', 1e18
    for lab, ref in tpl.items():
        d = np.mean((glyph.astype(float) - ref.astype(float)) ** 2)
        if d < bd:
            bd, best = d, lab
    return best


def classify_conf(glyph, tpl):
    """Return (letter, distance, margin) where margin = 2nd-best - best."""
    if glyph is None or not tpl:
        return '?', 1e18, 0.0
    ds = sorted((np.mean((glyph.astype(float) - ref.astype(float)) ** 2), lab)
                for lab, ref in tpl.items())
    best_d, best = ds[0]
    margin = (ds[1][0] - best_d) if len(ds) > 1 else 1e18
    return best, best_d, margin


# Distance above which a glyph is considered unreliable (kept as '?')
UNKNOWN_DIST = 4200.0
UNKNOWN_MARGIN = 60.0


def read_game(img, templates=None):
    """Read a full game state from a screenshot.

    Returns (board_lines, rack, info) where board_lines is 15 strings of
    letters/'.', rack is a string, and info holds geometry + uncertain cells.
    """
    if templates is None:
        templates = load_templates()
    if not templates:
        raise RuntimeError('no templates found (run build_templates.py)')
    bbox = detect_board(img)
    grid, glyphs = read_board(img, bbox)
    rack_glyphs = read_rack(img, bbox[3])

    lines = []
    cells = {}
    unknown = []
    for rr in range(SIZE):
        row = []
        for cc in range(SIZE):
            if (rr, cc) in glyphs:
                L, d, m = classify_conf(glyphs[(rr, cc)], templates)
                if d > UNKNOWN_DIST or m < UNKNOWN_MARGIN:
                    cells[(rr, cc)] = '?'
                    unknown.append((rr, cc, L, d, m))
                    row.append('?')
                else:
                    cells[(rr, cc)] = L
                    row.append(L)
            else:
                row.append('.')
        lines.append(''.join(row))

    rack = ''
    for i, g in enumerate(rack_glyphs):
        L, d, m = classify_conf(g, templates)
        if g is None or d > UNKNOWN_DIST or m < UNKNOWN_MARGIN:
            rack += '?'
        else:
            rack += L

    info = {'bbox': bbox, 'cells': cells, 'glyphs': glyphs,
            'rack_glyphs': rack_glyphs, 'unknown': unknown, 'templates': templates}
    return lines, rack, info