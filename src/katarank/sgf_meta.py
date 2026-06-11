"""
KataRank — SGF metadata extraction
===================================
Lightweight reader for SGF root-node properties. Used to populate
KAB2Output.metadata without round-tripping through the C++ engine.

Only header properties are read; move sequences are never parsed.
"""

import re
from typing import Dict

# SGF property → metadata key
_SGF_PROPS = {
    'PB': 'player_black',
    'PW': 'player_white',
    'BR': 'black_rank',
    'WR': 'white_rank',
    'DT': 'date',
    'RU': 'rules',
    'KM': 'komi',
    'RE': 'result',
    'SZ': 'board_size',
    'EV': 'event',
    'HA': 'handicap',
}

# (?<![A-Za-z]) keeps two-letter props from matching inside longer
# identifiers or move properties; value allows escaped ']'.
_PROP_RE = {
    prop: re.compile(r'(?<![A-Za-z])' + prop + r'\[((?:[^\]\\]|\\.)*)\]')
    for prop in _SGF_PROPS
}


def parse_sgf_metadata(text: str) -> Dict[str, str]:
    """Extract header metadata from SGF content. Missing props are omitted."""
    # Limit the search to the start of the file: root properties come
    # before the move sequence, and this avoids scanning huge game trees.
    head = text[:4096]
    meta: Dict[str, str] = {}
    for prop, key in _SGF_PROPS.items():
        m = _PROP_RE[prop].search(head)
        if m:
            val = m.group(1).replace('\\]', ']').replace('\\\\', '\\').strip()
            if val:
                meta[key] = val
    return meta


def read_sgf_metadata(path: str) -> Dict[str, str]:
    """Read header metadata from an SGF file. Returns {} on read errors."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return parse_sgf_metadata(f.read(4096))
    except OSError:
        return {}
