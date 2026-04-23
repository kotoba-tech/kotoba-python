"""SDK test conftest.

Prepends the SDK's ``src`` dir to ``sys.path`` so ``import kotoba`` works
without pip-installing the package. Used by all tests under this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
