from __future__ import annotations

import sys
from types import ModuleType


def ensure_numpy_core_alias() -> None:
    """
    Older environments ship numpy<2 which exposes `numpy.core` rather than the
    relocated `numpy._core` package introduced in numpy>=2.0. Some serialized
    joblib artifacts reference the new module path, leading to
    `ModuleNotFoundError: numpy._core` when loading on numpy<2. This helper
    aliases the legacy module so deserialization remains backward compatible.
    """
    if "numpy._core" in sys.modules:
        return

    try:
        import numpy as _np
    except Exception:
        return

    core: ModuleType | None = getattr(_np, "core", None)
    if core is None:
        return

    sys.modules.setdefault("numpy._core", core)

    # Mirror the common private submodules that numpy 2.0 exposes under _core.
    for name in ("_multiarray_umath", "_operand_flag_tests", "numerictypes"):
        submod = getattr(core, name, None)
        if submod is not None:
            sys.modules.setdefault(f"numpy._core.{name}", submod)
