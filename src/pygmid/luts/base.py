import os
from abc import ABC, abstractmethod
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class _BaseLUT(ABC, Mapping):
    """Base LUT implementing mapping protocol for Lookup class."""

    filename: str = ""
    device: InitVar[Optional[str]] = None
    lut_kwargs: InitVar[dict] = field(default={}, repr=False)
    data: Mapping = field(init=False, repr=False)
    extensions: str | Tuple[str, ...] = field(init=False, default=())

    @staticmethod
    def sweep_exists(filename: os.PathLike, **kwargs) -> bool:
        return Path(filename).exists()

    @staticmethod
    @abstractmethod
    def _save(filename: os.PathLike, data: Dict) -> None:
        """Save the LUT data to a file."""
        pass

    def __post_init__(self, *args, **kwargs):
        ext = Path(self.filename).suffix
        assert ext in self.extensions, f"Invalid extension: {ext}"

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
