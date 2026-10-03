"""Windows process-priority tiers for KataGo subprocesses.

Children inherit the launcher's priority class unless one is given
explicitly. The server's launcher ran at BELOW_NORMAL (Task Scheduler's
default priority 7 is the likely source), so every katago process ran
below any ordinary CPU-heavy job on the box; KataGo's search threads then starved, the GPU sat idle,
and interactive queries (ownership/variation) hung for 30s+.

Each spawn site now states its tier explicitly:

- INTERACTIVE  ABOVE_NORMAL — a user is waiting on the page
- ON_DEMAND    NORMAL       — single-game /review requests
- BATCH        BELOW_NORMAL — bulk -list runs, yield to everything

On non-Windows platforms all flags are 0 (no-op).
"""

import logging
import sys

logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    _CREATE_NO_WINDOW = 0x08000000
    _ABOVE_NORMAL = 0x00008000
    _NORMAL = 0x00000020
    _BELOW_NORMAL = 0x00004000
else:
    _CREATE_NO_WINDOW = _ABOVE_NORMAL = _NORMAL = _BELOW_NORMAL = 0

INTERACTIVE_FLAGS = _CREATE_NO_WINDOW | _ABOVE_NORMAL
ON_DEMAND_FLAGS = _CREATE_NO_WINDOW | _NORMAL
BATCH_FLAGS = _CREATE_NO_WINDOW | _BELOW_NORMAL


def raise_server_priority() -> None:
    """Lift the API server itself to ABOVE_NORMAL so request handling
    isn't starved either. Best-effort; Windows only."""
    if sys.platform != 'win32':
        return
    import ctypes
    kernel32 = ctypes.windll.kernel32
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), _ABOVE_NORMAL):
        logger.warning('SetPriorityClass(ABOVE_NORMAL) failed')
