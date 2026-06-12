"""
KataRank — Lightweight SGF move extractor
==========================================
Parses the main-line move sequence from an SGF string into the format
required by KataGo's analysis JSON protocol.

Only handles the main trunk of the game tree (no branches). Sufficient
for reviewed human games — variations/branches are not needed here.
"""

import re
from typing import Dict, List, Optional, Tuple

# GTP column letters — skips 'I' as per GTP standard
_GTP_COLS = 'ABCDEFGHJKLMNOPQRST'


def _sgf_to_gtp(sgf_coord: str, board_size: int = 19) -> Optional[str]:
    """Convert SGF coordinate (e.g. 'pd') to GTP notation (e.g. 'Q16').

    Returns 'pass' for empty-string coords (SGF pass) or 'tt' (old-style pass).
    Returns None for invalid coords.
    """
    if not sgf_coord or sgf_coord == 'tt':
        return 'pass'
    if len(sgf_coord) != 2:
        return None
    col_idx = ord(sgf_coord[0]) - ord('a')
    row_from_top = ord(sgf_coord[1]) - ord('a')
    if not (0 <= col_idx < board_size and 0 <= row_from_top < board_size):
        return None
    gtp_row = board_size - row_from_top
    return f'{_GTP_COLS[col_idx]}{gtp_row}'


def extract_moves_for_analysis(sgf: str) -> Optional[Dict]:
    """Parse an SGF string and return a dict ready for the KataGo analysis JSON protocol.

    Returns None if no moves are found.

    Returned dict keys:
        moves       — List of [color, gtp_coord] pairs, e.g. [['B','Q16'],['W','D4'],...]
        board_size  — int (default 19)
        komi        — float (default 6.5)
        rules       — str ('chinese' | 'japanese' | 'korean' | 'aga'), default 'chinese'
    """
    # Board size
    sz_m = re.search(r'SZ\[(\d+)\]', sgf)
    board_size = int(sz_m.group(1)) if sz_m else 19

    # Komi
    km_m = re.search(r'KM\[([\d.]+)\]', sgf)
    komi = float(km_m.group(1)) if km_m else 6.5

    # Rules — normalise to KataGo form
    ru_m = re.search(r'RU\[([^\]]+)\]', sgf, re.IGNORECASE)
    raw = ru_m.group(1).lower() if ru_m else ''
    if 'chinese' in raw:
        rules = 'chinese'
    elif 'japanese' in raw:
        rules = 'japanese'
    elif 'korean' in raw:
        rules = 'korean'
    elif 'aga' in raw:
        rules = 'aga'
    else:
        rules = 'chinese'

    # Main-line move sequence: find B[..] and W[..] in SGF order.
    # This regex captures the color and the coordinate (0-2 lowercase letters).
    moves: List[List[str]] = []
    for m in re.finditer(r';([BW])\[([a-s]{0,2})\]', sgf):
        color = m.group(1)
        coord = m.group(2)
        gtp = _sgf_to_gtp(coord, board_size)
        if gtp is not None:
            moves.append([color, gtp])

    if not moves:
        return None

    return {
        'moves': moves,
        'board_size': board_size,
        'komi': komi,
        'rules': rules,
    }
