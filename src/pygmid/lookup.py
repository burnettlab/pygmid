"""pygmid Lookup class"""

import glob
import logging
import os
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import prettytable
from auto_all import public
from scipy.constants import k as kB
from scipy.interpolate import interpn

from pygmid.luts import LUTS
from pygmid.utility.numerical import interp1

LOGGER = logging.getLogger(__name__)


@public
@dataclass
class Lookup:
    """PyGMID Lookup Class"""

    filename: InitVar[Optional[str]] = None
    device: InitVar[Optional[str]] = None
    lut_kwargs: InitVar[dict] = field(default={}, compare=False)
    _mode: int = field(init=False, default=1, repr=False, compare=False)

    @property
    def __DATA(self):
        if not hasattr(self, "data"):
            self.data = {}
        return self.data

    @__DATA.setter
    def __DATA(self, val: Mapping):
        if (filename := val.get("filename", None)) is not None:
            # Choose appropriate LUT subclass based on file extension
            val = LUTS[Path(filename).suffix](**val)
        self.data = val

    @property
    def __modefuncmap(self) -> Callable:
        f = {1: self._SimpleLK, 2: self._SimpleLK, 3: self._RatioVRatioLK}[self._mode]
        # if isinstance(self.__DATA, _H5LUT):
        #     f = h5open(f, cls_override=self.__DATA)
        return f

    @__modefuncmap.setter
    def __modefuncmap(self, args: Tuple[Union[str, List[str]], Union[str, List[str]]]):
        """
        Function to set lookup mode
            MODE1: output is single variable, variable arg is single
            MODE2: output is ratio, variable arg is single
            MODE3: output is ratio, variable arg is ratio

        Args:
            outkey: keywords (list) of output argument
            varkey: keywords (list) of variable argument

        Returns:
            mode (integer). Error if invalid mode selected
        """
        outkey, varkey = args
        out_ratio = isinstance(outkey, list) and len(outkey) > 1
        var_ratio = isinstance(varkey, list) and len(varkey) > 1
        if out_ratio and var_ratio:
            self._mode = 3
            LOGGER.debug(
                f"Output {'_'.join(outkey)} and Input {'_'.join(varkey)} sets lookup mode to 3"
            )
        elif out_ratio and (not var_ratio):
            self._mode = 2
            LOGGER.debug(
                f"Output {'_'.join(outkey)} and Input {varkey} sets lookup mode to 2"
            )
        elif (not out_ratio) and (not var_ratio):
            self._mode = 1
            LOGGER.debug(f"Output {outkey} and Input {varkey} sets lookup mode to 1")
        else:
            raise ValueError(
                "Invalid syntax or usage mode! Please check documentation."
            )

    def __post_init__(self, filename, device, lut_kwargs):
        """
        Setup the Lookup object

        Assigns loaded data and defaults
        to the DATA member variable

        Args:
            filename
        Kwargs:
            Keyword arguments can be used to
            set default values for the lookup
            function. METHOD sets the method
            used for interpolation at the end of lookup
            mode 3. pchip by default
        """
        kwargs = {
            k.upper(): v for k, v in lut_kwargs.items()
        }  # convert kwargs to upper
        self.__load(filename, device, **kwargs)
        self.__default = {
            "L": kwargs.get("L", min(self.__DATA["L"])),
            "VGS": kwargs.get("VGS", self.__DATA["VGS"]),
            "VDS": kwargs.get("VDS", max(self.__DATA["VDS"]) / 2),
            "VSB": kwargs.get("VSB", 0.0),
            "METHOD": kwargs.get("METHOD", "pchip"),
            "VGB": kwargs.get("VGB", None),
            "GM_ID": kwargs.get("GM_ID", None),
            "ID_W": kwargs.get("ID_W", None),
            "VDB": kwargs.get("VDB", None),
        }
        LOGGER.debug(
            "\n\t".join(
                ["Default lookup values:"]
                + list(map(lambda it: f"{it[0]}: {it[1]}", self.__default.items()))
            )
        )

    def __load(self, filename, device, **kwargs):
        """
        Function to load data from file

        Loads array data from file. Currently supports
        - .mat files
            .mat is parsed to convert MATLAB cell data into a dictionary of
            arrays. Data is loaded from value with first non-header key.
            Python interprets MATLAB cell structures as 1-D nests. Nested
            data is accessed and deep copied to member DATA variable.
        - .pkl files
        - .hdf5 files

        Args:
            filename

        Returns:
            LUT data structure when file type supported, None otherwise
        """
        if filename is None:
            techsweep_dir = os.path.expandvars(
                os.getenv("TECHSWEEP_DIR", "$PDK_ROOT/techsweeps")
            )
            assert os.path.isdir(
                techsweep_dir
            ), f"TECHSWEEP_DIR does not exist: {techsweep_dir}"
            filename = os.path.join(
                techsweep_dir,
                next(
                    chain.from_iterable(
                        map(
                            lambda ext: glob.iglob(f"*{ext}", root_dir=techsweep_dir),
                            [".h5", ".hdf5", ".mat", ".pkl"],
                        )
                    )
                ),
            )

        LOGGER.debug(f"Loading lookup table from {filename}")
        try:
            self.__DATA = dict(filename=filename, device=device, lut_kwargs=kwargs)
        except KeyError:
            LOGGER.exception(
                f"File not supported (only .mat, .pkl, .h5 and .hdf5): {filename}"
            )
            raise TypeError(
                f"File not supported (only .mat, .pkl, .h5 and .hdf5): {filename}"
            )

    def __contains__(self, key):
        return key.upper() in self.__DATA.keys() or any(
            isinstance(v, dict) and key.upper() in v for v in self.__DATA.values()
        )

    def __getitem__(self, key):
        """
        __getitem__ dunder method overwritten to allow convenient
        pseudo array access to member data. Returns a copy of the
        member array.
        """
        if key not in self:
            LOGGER.error(f"Lookup table does not contain key {key}")
            raise ValueError(f"Lookup table does not contain key {key}")

        if key.upper() in self.__DATA:
            return np.copy(self.__DATA[key.upper()])
        else:
            k = next(
                filter(
                    lambda x: isinstance(self.__DATA[x], dict)
                    and key.upper() in self.__DATA[x],
                    self.__DATA.keys(),
                )
            )
            return np.copy(self.__DATA[k][key])

    def __setitem__(self, key, value):
        """
        __setitem__ dunder method overwritten to allow convenient
        pseudo array access to member data. Sets the member data
        to the value passed.
        """
        if key not in self:
            LOGGER.error(f"Lookup table does not contain key {key}")
            raise ValueError(f"Lookup table does not contain key {key}")

        if key.upper() in self.__DATA:
            self.__DATA[key.upper()] = np.copy(value)
        else:
            k = next(
                filter(
                    lambda x: isinstance(self.__DATA[x], dict)
                    and key.upper() in self.__DATA[x],
                    self.__DATA.keys(),
                )
            )
            self.__DATA[k][key] = np.copy(value)

    def keys(self):
        """
        Alias for keys() function
        """
        return self.__DATA.keys()

    def values(self):
        """
        Alias for values() function
        """
        return self.__DATA.values()

    def items(self):
        """
        Alias for items() function
        """
        return self.__DATA.items()

    def lookup(self, out, **kwargs):
        """
        Alias for look_up() function
        """
        return self.look_up(out, **kwargs)

    def look_up(self, out, **kwargs):
        """
        Entry method for lookup functionality

        Sanitises input. Extracts the variable key as first key value pair
        in kwargs dict. Both the outkey and varkey are converted to lists.
        String is split based on _ character.

        Mode is determined and appropriate lookup function is called from
        modefuncmap dict

        Args:
            out: desired variable to be interpolated 'GM', 'ID' etc
            kwargs: keyword arguments (dict). First key-value pair is
                    variable argument

        Returns:
            y: interpolated data, [] if erroneous mode selected
        """
        outkeys = out.upper().split("_")
        varkeys, vararg = next(iter((kwargs.items()))) if kwargs else (None, None)
        varkeys = str(varkeys).upper().split("_")

        kwargs = {k.upper(): v for k, v in kwargs.items()}  # convert kwargs to upper
        defaultdict = {
            k: self.__default.get(k) for k in ["L", "VGS", "VDS", "VSB", "METHOD"]
        }
        pars = {
            k: kwargs.get(k, v) for k, v in defaultdict.items()
        }  # extracts parameters from kwargs

        # common kwargs for interpolating functions
        ipkwargs = {"bounds_error": False, "fill_value": None}

        LOGGER.debug(
            "\n\t".join(
                [
                    f"Looking up {'_'.join(outkeys)}",
                    f"From {'_'.join(varkeys)}: {vararg}",
                ]
                + list(map(lambda it: f"{it[0]}: {it[1]}", pars.items()))
            )
        )
        # appropriate lookup function is called with modefuncmap dict
        self.__modefuncmap = (outkeys, varkeys)
        return self.__modefuncmap(outkeys, varkeys, vararg, pars, **ipkwargs)

    def _SimpleLK(self, outkeys, varkeys, vararg, pars, **ipkwargs):
        """
        Lookup for Modes 1 and 2

        Args:
            outkeys: list of keys for desired output e.g ['GM', 'ID'] for 'GM_ID'
            varkeys: unused
            pars: dict containing L, VGS, VDS and VSB data
        Output:
            output: interpolated data specified by outkeys Squeezed to remove extra
                    dimensions
        """

        if len(outkeys) > 1:
            num, den = outkeys
            with np.errstate(divide="ignore", invalid="ignore"):
                ydata = self.__DATA[num] / self.__DATA[den]
                # nan causing issues with interpn extrapolation
                ydata[np.isnan(ydata)] *= 0.0
        else:
            outkey = outkeys[0]
            ydata = self.__DATA[outkey]

        points = (
            self.__DATA["L"],
            self.__DATA["VGS"],
            self.__DATA["VDS"],
            self.__DATA["VSB"],
        )
        xi_mesh = np.array(
            np.meshgrid(pars["L"], pars["VGS"], pars["VDS"], pars["VSB"], indexing="ij")
        )
        xi = np.rollaxis(xi_mesh, 0, 5)
        xi = xi.reshape(int(xi_mesh.size / 4), 4)

        output = interpn(points, ydata, xi, **ipkwargs).reshape(
            len(np.atleast_1d(pars["L"])),
            len(np.atleast_1d(pars["VGS"])),
            len(np.atleast_1d(pars["VDS"])),
            len(np.atleast_1d(pars["VSB"])),
        )

        # remove extra dimensions
        return np.squeeze(output)

    def _RatioVRatioLK(self, outkeys, varkeys, vararg, pars, **ipkwargs):
        """
        Lookup for Mode 3

        Args:
            outkeys: list of keys for desired output e.g ['GM', 'ID'] for 'GM_ID'
            varkeys: list of keys for ratio input e.g ['GM', 'ID'] for 'GM_ID'
            pars: dict containing L, VGS, VDS and VSB data
        Output:
            output: interpolated data specified by outkeys. Squeezed to remove extra
                    dimensions
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            # unpack outkeys and ydata
            num, den = outkeys
            ydata = self.__DATA[num] / self.__DATA[den]
            ydata[np.isnan(ydata)] *= 0.0
            # unpack varkeys and xdata
            num, den = varkeys
            xdata = self.__DATA[num] / self.__DATA[den]
            xdata[np.isnan(xdata)] *= 0.0

        xdesired = np.atleast_1d(vararg)

        points = (
            self.__DATA["L"],
            self.__DATA["VGS"],
            self.__DATA["VDS"],
            self.__DATA["VSB"],
        )
        xi_mesh = np.array(
            np.meshgrid(pars["L"], pars["VGS"], pars["VDS"], pars["VSB"], indexing="ij")
        )
        xi = np.rollaxis(xi_mesh, 0, 5)
        xi = xi.reshape(int(xi_mesh.size / 4), 4)

        x = interpn(points, xdata, xi, **ipkwargs).reshape(
            len(np.atleast_1d(pars["L"])),
            len(np.atleast_1d(pars["VGS"])),
            len(np.atleast_1d(pars["VDS"])),
            len(np.atleast_1d(pars["VSB"])),
        )

        y = interpn(points, ydata, xi, **ipkwargs).reshape(
            len(np.atleast_1d(pars["L"])),
            len(np.atleast_1d(pars["VGS"])),
            len(np.atleast_1d(pars["VDS"])),
            len(np.atleast_1d(pars["VSB"])),
        )

        x = np.atleast_2d(np.squeeze(np.transpose(x, (1, 0, 2, 3))))
        y = np.atleast_2d(np.squeeze(np.transpose(y, (1, 0, 2, 3))))
        if x.shape[0] == 1:
            x = np.moveaxis(x, 0, -1)
        if y.shape[0] == 1:
            y = np.moveaxis(y, 0, -1)

        dim = x.shape
        output = np.zeros((dim[1], len(xdesired)))  # type: ignore
        ipkwargs = {"kind": pars["METHOD"], "fill_value": np.nan}

        for i in range(0, dim[1]):
            for j in range(0, len(xdesired)):
                m = max(x[:, i])
                idx = np.argmax(x[:, i])
                if xdesired[j] > m:
                    LOGGER.warning(
                        f"Look up warning: {num}_{den} input larger than maximum! Output is NaN"
                    )
                if (num.upper() == "GM") and (den.upper() == "ID"):
                    x_right = x[idx:-1, i]
                    y_right = y[idx:-1, i]
                    output[i, j] = interp1(x_right, y_right, **ipkwargs)(xdesired[j])
                elif (
                    (num.upper() == "GM")
                    and (den.upper() == "CGG")
                    or (den.upper() == "CGG")
                ):
                    x_left = x[:idx, i]
                    y_left = y[:idx, i]
                    output[i, j] = interp1(x_left, y_left, **ipkwargs)(xdesired[j])
                else:
                    crossings = len(
                        np.argwhere(
                            np.diff(
                                np.sign(x[:, i] - xdesired[j] + np.finfo(float).eps)
                            )
                        )
                    )
                    if crossings > 1:
                        LOGGER.warning(">1 Crossings")
                        return []
                    output[i, j] = interp1(x[:, i], y[:, i], **ipkwargs)(xdesired[j])

        # remove extra dimensions
        return np.squeeze(output)

    def lookupVGS(self, **kwargs):
        return self.look_upVGS(**kwargs)

    def look_upVGS(self, **kwargs):
        """
        Companion function to "look_up." Finds transistor VGS for a given inversion level (GM_ID)
        or current density (ID/W) and given terminal voltages.
        The function interpolates (linear only) when the requested points lie off the simulation grid

        There are two basic usage scenarios:
        (1) Lookup VGS with known voltage at the source terminal
        (2) Lookup VGS with unknown source voltage, e.g. when the source of the
        transistor is the tail node of a differential pair

        At most one of the input arguments can be a vector; the other must be
        scalars.

        Examples of usage modes are given in test_lookupVGS.py

        Args:
            pars: dict containing L, VGB, GM_ID and ID_W, VDS, VSB and METHOD
        Output:
            output: 1-d numpy array
        """

        def perform_lk(self, **kwargs):
            kwargs = {
                k.upper(): v for k, v in kwargs.items()
            }  # convert kwargs to upper
            defaultdict = {
                k: self.__default.get(k)
                for k in ["L", "VDS", "VDB", "VGB", "GM_ID", "ID_W", "VSB", "METHOD"]
            }
            pars = {k: kwargs.pop(k, v) for k, v in defaultdict.items()}

            # Check whether GM_ID or ID_W was passed to function
            ratio_string = "None"
            ratio_data = None

            if pars["ID_W"] is not None:
                ratio_string = "ID_W"
                ratio_data = pars["ID_W"]

            elif pars["GM_ID"] is not None:
                ratio_string = "GM_ID"
                ratio_data = pars["GM_ID"]

            # determining the mode
            # In usage mode (1), the inputs to the function are GM_ID (or ID/W), L,
            # VDS and VSB
            if (pars["VGB"] and pars["VDB"]) is None:
                LOGGER.debug("Lookup VGS mode 1 (source-relative)")
                mode = 1
            # In usage mode (2), VDB and VGB must be supplied to the function
            elif (pars["VGB"] and pars["VDB"]) is not None:
                LOGGER.debug("Lookup VGS mode 2 (bulk-relative)")
                mode = 2
            else:
                LOGGER.error("Invalid lookup VGS usage!")
                raise SyntaxError("Invalid syntax or usage mode!")

            if mode == 1:
                VGS = self.__DATA["VGS"]
                ratio = self.look_up(
                    ratio_string,
                    VGS=VGS,
                    VDS=pars["VDS"],
                    VSB=pars["VSB"],
                    L=pars["L"],
                    **kwargs,
                )
            elif mode == 2:
                step = self.__DATA["VGS"][0] - self.__DATA["VGS"][1]
                VSB = np.arange(
                    max(self.__DATA["VSB"]), min(self.__DATA["VSB"]) + step, step
                )
                VGS = pars["VGB"] - VSB
                VDS = pars["VDB"] - VSB
                ratio = np.array(
                    [
                        self.look_up(
                            ratio_string,
                            VGS=VGS[i],
                            VDS=VDS[i],
                            VSB=VSB[i],
                            L=pars["L"],
                            **kwargs,
                        ).item()
                        for i in range(len(VGS))
                    ]
                )
                idx = ~np.isnan(ratio)
                ratio = ratio[idx]
                VGS = VGS[idx]
            else:
                raise RuntimeError("Invalid mode selected!")

            if np.size(pars["L"]) == 1:
                ratio.shape += (1,)
            else:
                ratio = np.swapaxes(ratio, 0, 1)

            s = ratio.shape

            output = np.empty((s[1], len(np.atleast_1d(ratio_data))))  # type: ignore
            output[:] = np.nan

            m = np.max(ratio)
            for j in range(s[1]):
                ratio_range = ratio[:, j]
                VGS_range = VGS

                if ratio_string == "GM_ID":
                    idx = np.where(ratio == m)[0].item()
                    VGS_range = VGS_range[idx:]
                    ratio_range = ratio_range[idx:]

                    if np.max(np.atleast_1d(ratio_data)) > m:  # type: ignore
                        LOGGER.warning("look_upVGS: GM_ID input larger than maximum!")

                output[j, :] = interp1(ratio_range, VGS_range)(ratio_data)
                output = output[:]

            return np.squeeze(output)

        # if isinstance(self.__DATA, _H5LUT):
        #     perform_lk = h5open(perform_lk, cls_override=self.__DATA)
        return perform_lk(self, **kwargs)

    def gamma(self, **kwargs):
        """
        Companion gamma function. Computes gamma from:

            STH/gm * 1/(4kT)

        where STH is thermal noise psd at 1 Hz

        Args:
            **kwargs: lookup parameters, GM_ID, length, VDS etc...
        Output:
            output: interpolated data specified by outkeys. Squeezed to remove extra
                    dimensions
        """
        # should provide a GMID, VDS and L
        return self.look_up("STH_GM", **kwargs) / (4 * kB * self["TEMP"].item())

    def fco(self, **kwargs):
        """
        Companion flicker corner function. Computes flicker corner from:

            SFL/STH

        where STH is thermal noise psd at 1 Hz
        and SFL is flicker noise psd at 1 Hz

        Args:
            **kwargs: lookup parameters, GM_ID, length, VDS etc...
        Output:
            output: interpolated data specified by outkeys. Squeezed to remove extra
                    dimensions
        """
        return self.look_up("SFL_STH", **kwargs)

    def __repr__(self) -> str:
        return f"PyGMID_Lookup<{self['INFO'].astype(str)}>"

    def __str__(self) -> str:
        tab = prettytable.PrettyTable()
        tab.title = f"PyGMID: {self['INFO'].astype(str)}"
        tab.field_names = ["Variable", "Size", "Min", "Max"]

        for k, v in filter(lambda it: hasattr(it[1], "dtype"), self.items()):
            is_numeric = np.issubdtype(v.dtype, np.number)
            size = (
                str(v.shape)
                .replace("(", "")
                .replace(")", "")
                .replace(", ", "x")
                .replace(",", "")
                if is_numeric
                else None
            )

            tab.add_row(
                [
                    k,
                    size if size else "1",
                    f"{v.min():.2e}" if v.size and is_numeric else "N/A",
                    f"{v.max():.2e}" if v.size and is_numeric else "N/A",
                ]
            )

        return f"PyGMID (from {self.__DATA}):\n{tab}"
