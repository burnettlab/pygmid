from collections import defaultdict
from typing import Callable, Tuple, Union

from auto_all import public
from blab_pyutils.types import NUM, Vector
from blab_pyutils.units import *
from blab_pyutils.units.utility import (
    apply_unit_wraps,
    arg_unit_conv,
    obj_using_units,
    return_unit_conv,
)

from pygmid.lookup import Lookup


@public
@apply_unit_wraps
class UnitLookup(Lookup):
    """Extension of PyGMID Lookup to add unit conversion functionality."""

    @arg_unit_conv
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.USE_UNITS = obj_using_units(self)

    @staticmethod
    def unit_func(
        key: str, *args, **kwargs
    ) -> Callable[[Union[str, NUM]], Union[str, NUM]]:
        unit_mapping: dict[str, Callable[[Union[str, NUM]], Union[str, NUM]]] = (
            defaultdict(lambda: lambda x: x)
        )
        unit_mapping.update(
            dict(
                CDD=Farad,
                CDG=Farad,
                CGB=Farad,
                CGD=Farad,
                CGG=Farad,
                CGS=Farad,
                CSG=Farad,
                CSS=Farad,
                GDS=Siemens,
                GM=Siemens,
                GMB=Siemens,
                ID=Ampere,
                IGD=Ampere,
                IGS=Ampere,
                L=Micron,
                LENGTH_PRECISION=Micron,
                WIDTH_PRECISION=Micron,
                MAX_LENGTH=Micron,
                MAX_WIDTH=Micron,
                MIN_LENGTH=Micron,
                MIN_WIDTH=Micron,
                W=Micron,
                SFL=lambda x: Ampere(x) * Ampere(1) / Hertz(1),
                STH=lambda x: Volt(x) * Volt(1) / Hertz(1),
                TEMP=lambda x: Kelvin(Kelvin(x)),
                VDD=Volt,
                VDS=Volt,
                VGS=Volt,
                VSB=Volt,
                VT=Volt,
            )
        )

        return unit_mapping[key.upper()]

    @staticmethod
    def ratio_units(
        key: Union[str, Tuple[str, str]],
    ) -> Callable[[Union[str, NUM]], Union[str, NUM]]:
        if isinstance(key, str):
            if "_" not in key:
                return UnitLookup.unit_func(key)

            num, den = key.split("_")
        else:
            if len(key) == 1:
                return UnitLookup.unit_func(key[0])

            num, den = key

        return lambda x: UnitLookup.unit_func(num)(x) / UnitLookup.unit_func(den)(1)  # type: ignore

    @return_unit_conv
    def __getitem__(self, key: str):
        return UnitLookup.unit_func(key)(super().__getitem__(key))

    def values(self):
        # Materialize values using __getitem__ so any conversion is applied
        return (self[k] for k in self.keys())

    def items(self):
        # Materialize (key, value) pairs using __getitem__ so conversion is applied
        return ((k, self[k]) for k in self.keys())

    @arg_unit_conv
    def lookup(self, out, **kwargs):
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
        return super().look_up(out, **kwargs)

    @arg_unit_conv
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
        return super().look_up(out, **kwargs)

    @arg_unit_conv
    def lookupVGS(self, **kwargs) -> Vector[Volt]:
        """Look up VGS from the model with unit conversion."""
        p_use = getattr(self, "USE_UNITS", None)
        self.USE_UNITS = False

        res = super().look_upVGS(**kwargs)

        self.USE_UNITS = p_use
        return res

    @arg_unit_conv
    def look_upVGS(self, **kwargs) -> Vector[Volt]:
        """Look up VGS from the model with unit conversion."""
        p_use = getattr(self, "USE_UNITS", None)
        self.USE_UNITS = False

        res = super().look_upVGS(**kwargs)

        self.USE_UNITS = p_use
        return res

    def gamma(self, **kwargs) -> UREG.ampere**2 / UREG.Hz:
        """
        Companion gamma function. Computes gamma from:

            STH/gm * 1/(4kT)

        where STH is thermal noise psd at 1 Hertz

        Args:
            **kwargs: lookup parameters, GM_ID, length, VDS etc...
        Output:
            output: interpolated data specified by outkeys. Squeezed to remove extra
                    dimensions
        """
        # should provide a GMID, VDS and L
        return super().gamma(**kwargs)

    def fco(self, **kwargs) -> Hertz:
        """
        Companion flicker corner function. Computes flicker corner from:

            SFL/STH

        where STH is thermal noise psd at 1 Hertz
        and SFL is flicker noise psd at 1 Hertz

        Args:
            **kwargs: lookup parameters, GM_ID, length, VDS etc...
        Output:
            output: interpolated data specified by outkeys. Squeezed to remove extra
                    dimensions
        """
        return super().fco(**kwargs)

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
        USING_UNITS = obj_using_units(self)
        res = super()._SimpleLK(outkeys, varkeys, vararg, pars, **ipkwargs)
        return UnitLookup.ratio_units(outkeys)(res) if USING_UNITS else res

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
        USING_UNITS = obj_using_units(self)
        res = super()._RatioVRatioLK(outkeys, varkeys, vararg, pars, **ipkwargs)
        return UnitLookup.ratio_units(outkeys)(res) if USING_UNITS else res

    def __repr__(self):
        return f"UnitLUT({super().__repr__()})"
