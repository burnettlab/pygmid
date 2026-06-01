import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, KeysView, Optional, Tuple

import h5py
import numpy as np
from auto_all import public

from pygmid.utility.numerical import convert_temp, num_conv

from .base import _BaseLUT


def get_group_name(**kwargs) -> str:
    return "/".join(
        map(
            lambda k: f"{k}:{kwargs[k]}" if k not in ["DEVICE"] else str(kwargs[k]),
            ["CORNER", "TEMP", "VDD", "DEVICE"],
        )
    )


def h5open(func: Optional[Callable] = None, *, cls_override: Optional[Any] = None):
    if func is None:
        return partial(h5open, cls_override=cls_override)

    @wraps(func)
    def open_h5(cls, *args, **kwargs):
        c = cls_override or cls
        # consider the file "not opened" unless it's an h5py.File instance
        if closing := not isinstance(c._h5file, h5py.File):
            c._h5file = h5py.File(c.filename, "r")

        assert isinstance(c._h5file, h5py.File), "HDF5 file not opened"
        res = func(cls, *args, **kwargs)

        if closing:
            c._h5file.close()
            c._h5file = None
        return res

    return open_h5


@public
@dataclass
class _H5LUT(_BaseLUT):
    device: Optional[str]
    _h5file: Optional[h5py.File] = field(default=None, repr=False)
    extensions: Tuple[str, ...] = (".h5", ".hdf5")

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.env_kwargs = {k.upper(): v for k, v in kwargs.items()}

    @staticmethod
    def sweep_exists(filename: os.PathLike, **kwargs) -> bool:
        if not _BaseLUT.sweep_exists(filename, **kwargs):
            return False

        with h5py.File(Path(filename), "r") as f:
            return get_group_name(**kwargs) in f

    @staticmethod
    def _save(filename: os.PathLike, data: Dict) -> None:
        with h5py.File(filename, "a") as f:
            if (group_name := get_group_name(**data)) in f:
                grp = f[group_name]
            else:
                grp = f.create_group(group_name)

            for key, value in data.items():
                if key in grp:
                    grp[key] = value
                else:
                    grp.create_dataset(key, data=value)

    @property
    def env_kwargs(self):
        if not hasattr(self, "_env_kwargs"):
            self.env_kwargs = {}
        return self._env_kwargs

    @env_kwargs.setter
    def env_kwargs(self, val: dict):
        default = {
            "CORNER": "NOM",
            "TEMP": "room",
        }
        default.update({k.upper(): v for k, v in val.items()})
        self._env_kwargs = default
        try:
            delattr(self, "lut_key")
        except AttributeError:
            pass

    @cached_property
    @h5open
    def lut_key(self) -> str:
        """Open the HDF5 file and resolve the final group for the given environment/device.
        Returns the h5 group object name.
        """
        grp = self._h5file

        def key_conv(e):
            return convert_temp(e, temp_unit="K") if k == "TEMP" else num_conv(e)

        def dist_calc(x):
            return (
                abs(x - env_val)
                if x <= env_val or k != "VDD" or np.isclose(x, env_val)
                else float("inf")
            )

        # traverse environment keys that are of the form KEY:val
        while len(env_keys := set(map(lambda k: k.split(":")[0], grp.keys()))) == 1:  # type: ignore
            k = next(iter(env_keys))
            grp_keys = list(k.split(":")[1] for k in grp.keys())  # type: ignore

            env_val = key_conv(
                self.env_kwargs.get(
                    k.upper(), globals().get(k.upper(), os.getenv(k.upper()))
                )
            )
            assert env_val is not None, f"Environment variable {k} not specified!"

            if isinstance(env_val, str):
                chosen = env_val
            else:
                chosen = grp_keys[
                    np.argmin([dist_calc(key_conv(ck)) for ck in grp_keys])
                ]

            grp = grp[f"{k}:{chosen}"]

        # Load by device
        if self.device is None and set(grp.keys()) == {"n", "p"}:  # type: ignore
            raise ValueError(
                "Device type must be specified when both n and p data are present in the file."
            )
        elif self.device is not None:
            grp = grp[self.device]

        return grp.name  # type: ignore

    @h5open
    def keys(self) -> KeysView[Any]:
        return iter(list(self._h5file[self.lut_key].keys()))  # type: ignore

    @h5open
    def values(self):
        return iter(list(self[k] for k in self._h5file[self.lut_key].keys()))  # type: ignore

    @h5open
    def items(self):
        return iter(list((k, self[k]) for k in self._h5file[self.lut_key].keys()))  # type: ignore

    @h5open
    def __getitem__(self, key) -> Any:
        k = key.upper()
        item = self._h5file[self.lut_key][k]
        # Some HDF5 objects (datasets) support the [()] shorthand to read all
        # data, but in some files the retrieved object may be a structured
        # dtype Field or other non-subscriptable object. Try the common
        # access patterns and fall back to a safe deepcopy.
        try:
            data = item[()]
        except TypeError:
            # Not subscriptable — attempt to convert or deepcopy directly
            try:
                data = np.array(item)
            except Exception:
                try:
                    data = deepcopy(item)
                except Exception:
                    # Last resort: return the item as-is
                    data = item
        return deepcopy(data)

    def __str__(self) -> str:
        return f"{super().__str__()}{self.lut_key}"
