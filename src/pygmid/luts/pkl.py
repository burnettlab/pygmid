import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

from auto_all import public

from .base import _BaseLUT


@public
@dataclass
class _PKLLUT(_BaseLUT):
    extensions: str = ".pkl"

    @staticmethod
    def _save(filename: os.PathLike, data: Dict) -> None:
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with open(self.filename, "rb") as f:
            data = pickle.load(f)
        # normalize keys to upper
        self.data = {k.upper(): v for k, v in data.items()}

    def __getitem__(self, key):
        k = key.upper()
        val = self.data[k]
        return (
            deepcopy(val)
            if not isinstance(val, dict)
            else {kk: deepcopy(vv) for kk, vv in val.items()}
        )
