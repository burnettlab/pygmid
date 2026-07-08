import os
from abc import ABC, abstractmethod
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from dataclasses import InitVar, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class _BaseLUT(ABC, Mapping):
    """Base LUT implementing mapping protocol for Lookup class."""

    filename: str = ""
    device: InitVar[Optional[str]] = None
    lut_kwargs: InitVar[dict] = field(default={}, repr=False)
    data: Mapping = field(init=False, repr=False)
    extensions: str | Tuple[str, ...] = field(init=False, default=())

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        wrapped_by = getattr(cls.__getitem__, "_wrapped", set())
        if "wrapped_getitem" in wrapped_by:
            return

        original_getitem = cls.__getitem__

        @wraps(original_getitem)
        def wrapped_getitem(self, key):
            val = original_getitem(self, key)
            return self._decode_bytes(val)

        cls.__getitem__ = wrapped_getitem
        wrapped_by.add("wrapped_getitem")
        setattr(cls.__getitem__, "_wrapped", wrapped_by)

    def __post_init__(self, *args, **kwargs):
        ext = Path(self.filename).suffix
        assert ext in self.extensions, f"Invalid extension: {ext}"

    @staticmethod
    def sweep_exists(filename: os.PathLike, **kwargs) -> bool:
        return Path(filename).exists()

    @staticmethod
    @abstractmethod
    def _save(filename: os.PathLike, data: Dict) -> None:
        """Save the LUT data to a file."""
        pass

    def _decode_bytes(self, val: Any) -> Any:
        """
        Recursively converts byte strings to python strings and
        expands environment variables in potential file paths.
        """

        def transform_item(item):
            # 1. Convert bytes to string
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="ignore")

            # 2. Expand environment variables if it's a string
            # os.path.expandvars is safe to call on non-path strings;
            # it returns them unchanged if no variables are found.
            if isinstance(item, str):
                item = os.path.expandvars(item)

            # 3. Handle nested structures
            if isinstance(item, np.ndarray) and item.dtype.kind == "S":
                return item.astype(
                    str
                )  # Note: apply expandvars if this is then iterated
            if isinstance(item, (dict, Mapping)):
                return {k: self._decode_bytes(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return type(item)(self._decode_bytes(v) for v in item)

            return item

        if isinstance(val, np.ndarray):
            if val.dtype.kind == "S":
                # For byte arrays, convert to string first, then expand vars element-wise
                strings = val.astype(str)
                vfunc = np.vectorize(lambda x: os.path.expandvars(x))
                return vfunc(strings)
            if val.dtype == object:
                vfunc = np.vectorize(transform_item)
                return vfunc(val)

        return transform_item(val)

    def __contains__(self, key):
        k = key.upper()
        iters = [self.keys()]
        while iters:
            current_iter = iters.pop()
            for item_key in current_iter:
                if item_key == k:
                    return True
                if isinstance(self[item_key], (dict, Mapping)):
                    iters.append(self[item_key].keys())

        return False

    def __iter__(self):
        for k in self.keys():
            yield k

    def __len__(self):
        return len(list(self.keys()))

    def keys(self) -> KeysView[Any]:
        return self.data.keys()

    def values(self) -> ValuesView[Any]:
        return self.data.values()

    def items(self) -> ItemsView[Any, Any]:
        return self.data.items()

    def __getstate__(self):
        return dict(filter(lambda it: not it[0].startswith("_"), self.__dict__.items()))
        state = self.__dict__.copy()
        # Remove unpicklable entries
        if "_h5file" in state:
            del state["_h5file"]
        return state

    def __str__(self) -> str:
        return f"filename={self.filename}"
