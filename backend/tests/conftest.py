"""Pytest bootstrap for the Phase 2 backend tests.

Adds the ``backend/`` directory to ``sys.path`` so ``import app.*`` resolves no
matter which directory ``pytest`` is launched from.
"""

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
