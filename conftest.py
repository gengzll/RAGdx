"""Test bootstrap — make the in-tree ``src/`` layout importable without an editable install.

When ragdx is installed via ``pip install -e .`` this file is harmless. It only
matters for environments where the package is not installed and tests are run
directly from the checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
