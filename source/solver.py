"""
Scrabble move generator + scorer.

Board format: list of 15 strings, each 15 chars. '.' = empty, uppercase = tile,
lowercase = blank played as that letter (scores 0).

Rack: string like 'IOOTZOU' or 'IOOTZ?U' ('?' = blank).

Usage:
    python solver.py board.txt RACK [dict.txt] [--top N]
"""
import os
import sys
import time

SIZE = 15
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

LETTER_VALUES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10,
}

TW = {(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)}
DW = {(1, 1), (1, 13), (2, 2), (2, 12), (3, 3), (3, 11), (4, 4), (4, 10),
      (10, 4), (10, 10), (11, 3), (11, 11), (12, 2), (12, 12), (13, 1),
      (13, 13), (7, 7)}
TL = {(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13), (9, 1), (9, 5),
      (9, 9), (9, 13), (13, 5), (13, 9)}
DL = {(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2),
      (6, 6), (6, 8), (6, 12), (7, 3), (7, 11), (8, 2), (8, 6), (8, 8),
      (8, 12), (11, 0), (11, 7), (11, 14), (12, 6), (12, 8), (14, 3),
      (14, 11)}


def build_trie(words):
    root = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node['#'] = True
    return root


def load_words(path):
    out = set()
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            w = line.strip().upper()
            if w.isalpha():
                out.add(w)
    return out


def premium(r, c):
    """Return (letter_mult, word_mult)."""
    if (r, c) in TW:
        return 1, 3
    if (r, c) in DW:
        return 1, 2
    if (r, c) in TL:
        return 3, 1
    if (r, c) in DL:
        return 2, 1
    return 1, 1


class Board:
    def __init__(self, lines=None):
        if lines is None:
            self.g = [['.'] * SIZE for _ in range(SIZE)]
        else:
            self.g = [list(row) for row in lines]
        self.empty = all(ch == '.' for row in self.g for ch in row)

    def occupied(self, r, c):
        return 0 <= r < SIZE and 0 <= c < SIZE and self.g[r][c] != '.'

    def anchors(self):
        if self.empty:
            return {(7, 7)}
        out = set()
        for r in range(SIZE):
            for c in range(SIZE):
                if self.g[r][c] != '.':
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if self.occupied(nr, nc):
                        out.add((r, c))
                        break
        return out

    def cross_allowed(self, r, c, direction):
        """Letters allowed at empty (r,c) so the perpendicular word is valid."""
        if direction == 'H':      # main word horizontal -> perpendicular is vertical
            top = r
            while top - 1 >= 0 and self.g[top - 1][c] != '.':
                top -= 1
            bot = r
            while bot + 1 < SIZE and self.g[bot + 1][c] != '.':
                bot += 1
            pre = ''.join(self.g[i][c] for i in range(top, r)).upper()
            suf = ''.join(self.g[i][c] for i in range(r + 1, bot + 1)).upper()
        else:                     # main vertical -> perpendicular horizontal
            left = c
            while left - 1 >= 0 and self.g[r][left - 1] != '.':
                left -= 1
            right = c
            while right + 1 < SIZE and self.g[r][right + 1] != '.':
                right += 1
            pre = ''.join(self.g[r][j] for j in range(left, c)).upper()
            suf = ''.join(self.g[r][j] for j in range(c + 1, right + 1)).upper()
        if pre == '' and suf == '':
            return None  # any letter fine
        return {L for L in ALPHABET if (pre + L + suf) in WORDS}


WORDS = set()


def score_play(grid, placed, blanks):
    """grid: 15x15 letters (blank letters lowercase). placed: set of (r,c).

    Only words that contain at least one NEWLY placed tile score. Pre-existing
    words elsewhere on the board must not be counted (doing so inflated every
    candidate by the same amount and corrupted the move ranking).
    """
    total = 0
    words = []
    for r in range(SIZE):
        c = 0
        while c < SIZE:
            if grid[r][c] != '.':
                s = c
                while c < SIZE and grid[r][c] != '.':
                    c += 1
                if c - s >= 2:
                    words.append([(r, x) for x in range(s, c)])
            else:
                c += 1
    for c in range(SIZE):
        r = 0
        while r < SIZE:
            if grid[r][c] != '.':
                s = r
                while r < SIZE and grid[r][c] != '.':
                    r += 1
                if r - s >= 2:
                    words.append([(x, c) for x in range(s, r)])
            else:
                r += 1
    scored = 0
    for w in words:
        if not any((r, c) in placed for (r, c) in w):
            continue          # word untouched by this play -> does not score
        ws = 0
        wm = 1
        for (r, c) in w:
            ch = grid[r][c]
            if ch.isupper():
                val = LETTER_VALUES.get(ch, 0)
            else:
                val = 0
            if (r, c) in placed:
                lm, wmul = premium(r, c)
                val *= lm
                wm *= wmul
            ws += val
        total += ws * wm
        scored += 1
    if len(placed) == 7 and total > 0:
        total += 50            # bingo bonus
    return total, scored


def generate(board, rack):
    anchors = board.anchors()
    moves = {}  # (start,r,c,dir,word) -> placed info
    grid = [row[:] for row in board.g]

    for direction in ('H', 'V'):
        for li in range(SIZE):
            def cell(j):
                return (li, j) if direction == 'H' else (j, li)

            def occ(j):
                r, c = cell(j)
                return board.g[r][c] != '.'

            def letters_at(j):
                r, c = cell(j)
                return board.g[r][c].upper()

            def rec(pos, node, rack_left, placed, has_existing, on_anchor, start):
                if pos >= SIZE:
                    return
                if occ(pos):
                    ch = letters_at(pos)
                    nxt = node.get(ch)
                    if nxt is not None:
                        rec(pos + 1, nxt, rack_left, placed, True, on_anchor, start)
                    return
                # empty cell
                allowed = board.cross_allowed(*cell(pos), direction)
                # place a tile
                tried = set()
                for i, ch in enumerate(rack_left):
                    if ch in tried:
                        continue
                    tried.add(ch)
                    L = ch
                    if L == '?':
                        continue
                    if allowed is not None and L not in allowed:
                        continue
                    nxt = node.get(L)
                    if nxt is None:
                        continue
                    r, c = cell(pos)
                    on_anc = on_anchor or (r, c) in anchors
                    new_placed = placed + [(r, c, L)]
                    rec(pos + 1, nxt, rack_left[:i] + rack_left[i + 1:],
                        new_placed, has_existing, on_anc, start)
                # blank tile
                if '?' in rack_left:
                    idx = rack_left.index('?')
                    for L in ALPHABET:
                        if allowed is not None and L not in allowed:
                            continue
                        nxt = node.get(L)
                        if nxt is None:
                            continue
                        r, c = cell(pos)
                        on_anc = on_anchor or (r, c) in anchors
                        new_placed = placed + [(r, c, L.lower())]
                        rec(pos + 1, nxt, rack_left[:idx] + rack_left[idx + 1:],
                            new_placed, has_existing, on_anc, start)
                # stop the word here
                if placed and node.get('#') and (has_existing or on_anchor):
                    add_move(start, pos - 1, placed, direction)

            def add_move(start, end, placed, direction):
                word = ''.join(
                    (board.g[r][c].upper() if board.g[r][c] != '.' else L.upper())
                    for (r, c, L) in placed
                )
                # build full word spanning start..end
                if direction == 'H':
                    r = li
                    full = ''.join(board.g[r][x].upper() if board.g[r][x] != '.'
                                   else [p[2].upper() for p in placed
                                         if p[0] == r and p[1] == x][0]
                                   for x in range(start, end + 1))
                else:
                    c = li
                    full = ''.join(board.g[x][c].upper() if board.g[x][c] != '.'
                                   else [p[2].upper() for p in placed
                                         if p[1] == c and p[0] == x][0]
                                   for x in range(start, end + 1))
                # apply to grid copy and score
                g2 = [row[:] for row in board.g]
                newpos = set()
                blanks = 0
                for (r, c, L) in placed:
                    g2[r][c] = L
                    newpos.add((r, c))
                    if L.islower():
                        blanks += 1
                sc, nwords = score_play(g2, newpos, blanks)
                if sc <= 0:
                    return
                key = (direction, start, end, li)
                if key not in moves or moves[key][0] < sc:
                    moves[key] = (sc, full, tuple(placed), len(placed),
                                  tuple(cell(start)), direction)

            # try each start where left neighbour is empty/edge
            for s in range(SIZE):
                if s > 0 and occ(s - 1):
                    continue
                rec(s, ROOT, rack, [], False, False, s)

    results = []
    for key, (sc, full, placed, n, start_cell, d) in moves.items():
        results.append((sc, full, placed, start_cell, d))
    results.sort(key=lambda x: (-x[0], -len(x[2])))
    return results


ROOT = {}

# The NWL2023 dictionary is ~196k words. Loading it and building the trie on
# every solve() call cost ~1s and churned a fresh multi-hundred-MB trie each
# time (GC pressure + wasted work, worst in "Watch" mode). Load it once per
# dictionary file and reuse it; the cache is keyed on path+mtime so an edited
# dictionary is still picked up.
_DICT_CACHE = {}


def _load_dict(wordfile):
    try:
        key = (os.path.abspath(wordfile), os.path.getmtime(wordfile))
    except OSError:
        key = (wordfile, None)
    entry = _DICT_CACHE.get(key)
    if entry is None:
        words = load_words(wordfile)
        entry = (words, build_trie(words))
        _DICT_CACHE.clear()          # only ever hold the active dictionary
        _DICT_CACHE[key] = entry
    return entry


def solve(board_lines, rack, wordfile='dict/nwl2023.txt'):
    global WORDS, ROOT
    WORDS, ROOT = _load_dict(wordfile)
    b = Board(board_lines)
    t = time.time()
    res = generate(b, rack.upper())
    return res, time.time() - t


def coord(r, c):
    return f"{'ABCDEFGHIJKLMNO'[c]}{r + 1}"


if __name__ == '__main__':
    args = sys.argv[1:]
    top = 20
    if '--top' in args:
        i = args.index('--top')
        top = int(args[i + 1])
        del args[i:i + 2]
    board_file = args[0]
    rack = args[1]
    wordfile = args[2] if len(args) > 2 else 'dict/nwl2023.txt'
    lines = [l.strip().upper().replace(' ', '.') for l in open(board_file)
             if l.strip()]
    lines = [l.rstrip() for l in lines]
    res, dt = solve(lines, rack, wordfile)
    print(f"dict={wordfile} rack={rack} moves={len(res)} time={dt:.2f}s")
    for sc, full, placed, start_cell, d in res[:top]:
        pos = ', '.join(f"{coord(r, c)}={L}" for (r, c, L) in placed)
        print(f"{sc:>4}  {full:<12} {d} at {coord(*start_cell)}  [{pos}]")
