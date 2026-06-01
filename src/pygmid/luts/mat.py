import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.io
from auto_all import public

from pygmid.luts.base import _BaseLUT


@public
@dataclass
class _MATLUT(_BaseLUT):
    extensions: str = ".mat"

    @staticmethod
    def _save(filename: os.PathLike, data: Dict) -> None:
        scipy.io.savemat(filename, data)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        mat = scipy.io.loadmat(self.filename, matlab_compatible=True)
        # find first non-header key
        for k in mat.keys():
            if not (k.startswith("__") and k.endswith("__")):
                mat_struct = mat[k]
                break
        else:
            raise RuntimeError("No valid data found in .mat file")

        # MATLAB struct array nesting: take first element
        # mat_struct is a numpy structured array
        self.data = {
            k.upper(): deepcopy(np.squeeze(mat_struct[k][0][0]))
            for k in mat_struct.dtype.names
        }

    def __getitem__(self, key):
        k = key.upper()
        val = self.data[k]
        return (
            deepcopy(val)
            if not isinstance(val, dict)
            else {kk: deepcopy(vv) for kk, vv in val.items()}
        )
