"""Pure-Python stand-in for pandas._libs.indexing (its compiled .pyd is blocked by Windows application control on this
machine). The blocked binary is never loaded; this module provides the single class pandas needs from it."""
import sys, types

_m = types.ModuleType("pandas._libs.indexing")

class NDFrameIndexerBase:
    def __init__(self, name, obj):
        self.obj = obj
        self.name = name
        self._ndim = None

    @property
    def ndim(self):
        ndim = self._ndim
        if ndim is None:
            ndim = self._ndim = self.obj.ndim
            if ndim > 2:
                raise ValueError("NDFrameIndexer does not support NDFrame objects with ndim > 2")
        return ndim

_m.NDFrameIndexerBase = NDFrameIndexerBase
sys.modules.setdefault("pandas._libs.indexing", _m)
