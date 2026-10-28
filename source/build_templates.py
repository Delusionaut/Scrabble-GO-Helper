"""Build letter templates from a labeled screenshot, then self-test.

The board glyphs below were read by eye from screen_raw.png.  They are averaged
per letter to form a clean template set.  J and W do not appear on this board,
so they are rendered from a system font as a fallback (isolated enough shapes).
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vision

IMG = r'C:\Users\Windows\scrabble_proj\screen_raw.png'

# (row, col) -> letter, read from the screenshot via zoomed crops
TRANS = {
    (4, 10): 'R', (5, 10): 'A', (5, 13): 'C', (6, 8): 'M', (6, 9): 'E', (6, 10): 'N',
    (6, 13): 'I', (7, 7): 'N', (7, 8): 'A', (7, 11): 'S', (7, 12): 'I', (7, 13): 'R',
    (8, 2): 'B', (8, 3): 'U', (8, 4): 'S', (8, 6): 'H', (8, 7): 'O', (8, 8): 'Y',
    (8, 9): 'A', (8, 11): 'I', (8, 13): 'Q', (9, 4): 'E', (9, 5): 'H', (9, 7): 'I',
    (9, 9): 'M', (9, 10): 'U', (9, 11): 'N', (9, 13): 'U', (10, 0): 'F', (10, 2): 'F',
    (10, 3): 'I', (10, 4): 'X', (10, 5): 'E', (10, 6): 'R', (10, 7): 'S', (10, 11): 'G',
    (10, 12): 'L', (10, 13): 'E', (10, 14): 'E', (11, 0): 'A', (11, 1): 'K', (11, 2): 'I',
    (11, 3): 'N', (11, 6): 'A', (11, 7): 'Y', (11, 11): 'L', (12, 0): 'N', (12, 2): 'B',
    (12, 4): 'T', (12, 5): 'A', (12, 6): 'G', (12, 10): 'V', (12, 11): 'E', (12, 12): 'G',
    (13, 0): 'E', (13, 4): 'A', (13, 5): 'D', (13, 11): 'D', (13, 12): 'O', (13, 13): 'N',
    (14, 0): 'S', (14, 3): 'T', (14, 4): 'R', (14, 5): 'O', (14, 6): 'P', (14, 7): 'E',
}
# rack order left->right: I O O T Z O U
RACK = 'IOOTZOU'

GLYPH = vision.GLYPH


def normalize_ink(th):
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = None
    best_area = 0
    h = th.shape[0]
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if bh < h * 0.35 or a < 30:
            continue
        if a > best_area:
            best_area, best = a, (bx, by, bw, bh)
    if best is None:
        return None
    bx, by, bw, bh = best
    g = th[by:by + bh, bx:bx + bw]
    s = float(GLYPH - 4) / max(bw, bh)
    nw, nh = max(1, int(round(bw * s))), max(1, int(round(bh * s)))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((GLYPH, GLYPH), np.uint8)
    ox, oy = (GLYPH - nw) // 2, (GLYPH - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = g
    return canvas


def render_letter(ch, font_path, size=200):
    font = ImageFont.truetype(font_path, size)
    im = Image.new('L', (size * 2, size * 2), 0)
    d = ImageDraw.Draw(im)
    d.text((size // 2, size // 2), ch, fill=255, font=font)
    arr = np.array(im)
    th = (arr > 128).astype(np.uint8) * 255
    return normalize_ink(th)


def main():
    img = cv2.imread(IMG)
    bbox = vision.detect_board(img)
    grid, glyphs = vision.read_board(img, bbox)
    rack = vision.read_rack(img, bbox[3])

    per_letter = {}
    for (rc, L) in TRANS.items():
        g = glyphs.get(rc)
        if g is None:
            print('WARN missing glyph', rc, L)
            continue
        per_letter.setdefault(L, []).append(g.astype(np.float32))
    for i, L in enumerate(RACK):
        g = rack[i] if i < len(rack) else None
        if g is None:
            print('WARN missing rack glyph', i, L)
            continue
        per_letter.setdefault(L, []).append(g.astype(np.float32))

    print('letters with samples:', {k: len(v) for k, v in sorted(per_letter.items())})

    tpl = {L: np.clip(np.mean(v, axis=0), 0, 255).astype(np.uint8)
           for L, v in per_letter.items()}

    # pick best font for J/W by scoring rendered known letters
    import glob
    fonts = glob.glob(r'C:\Windows\Fonts\*bd*.ttf') + glob.glob(r'C:\Windows\Fonts\arial*.ttf') \
        + glob.glob(r'C:\Windows\Fonts\segoeui*.ttf') + glob.glob(r'C:\Windows\Fonts\verdanab.ttf')
    best_font, best_err = None, 1e18
    for f in set(fonts):
        try:
            errs = []
            for L, ref in tpl.items():
                r = render_letter(L, f)
                if r is None:
                    continue
                errs.append(np.mean((r.astype(float) - ref.astype(float)) ** 2))
            if errs:
                e = float(np.mean(errs))
                if e < best_err:
                    best_err, best_font = e, f
        except Exception as ex:
            print('font skip', f, ex)
    print('best font for J/W:', best_font, 'mean err', round(best_err, 1))

    for L in 'JW':
        if L not in tpl and best_font:
            r = render_letter(L, best_font)
            if r is not None:
                tpl[L] = r

    vision.save_templates(tpl, r'C:\Users\Windows\scrabble_proj\templates.npz')
    print('saved templates:', ''.join(sorted(tpl.keys())))

    # self-test on the board
    bad = 0
    for (rc, L) in sorted(TRANS.items()):
        g = glyphs.get(rc)
        pred = vision.classify(g, tpl)
        if pred != L:
            bad += 1
            print('MISS', rc, 'true', L, 'pred', pred)
    print(f'self-test: {len(TRANS) - bad}/{len(TRANS)} correct')
    # worst-case margin report
    for (rc, L) in sorted(TRANS.items()):
        g = glyphs.get(rc)
        ds = sorted(((np.mean((g.astype(float) - ref.astype(float)) ** 2), lab)
                     for lab, ref in tpl.items()))
        if ds[0][1] != L or ds[1][0] - ds[0][0] < 200:
            pass
    print('done')


if __name__ == '__main__':
    main()