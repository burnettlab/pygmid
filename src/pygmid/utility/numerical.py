import numpy as np
from auto_all import public
from scipy.interpolate import PchipInterpolator, interp1d


@public
def monotonic_interp1(x, y, **ipkwargs):
    # check for maximum monotonic subarray here
    # ...
    # placeholder
    xsub = x
    ysub = y
    # ...

    return interp1(xsub, ysub, **ipkwargs)


@public
def interp1(x, y, **ipkwargs):
    """
    Wrapper function for python 1d interpolation
    Combines the functionality of interp1d and PchipInterpolator.

    Reorders x and y for increasing monotonicity in x

    Args:
        x, y
    Kwargs:
        Interpolation keywords - see interp1d
    """

    METHOD = ipkwargs.get("kind", "pchip")
    if METHOD == "pchip":
        # need to convert ipkwargs
        pchipkwargs = {
            "axis": ipkwargs.get("axis", 0),
            "extrapolate": ipkwargs.get("extrapolate", True),
        }
        # enforce increasing monotonicity
        ind = np.argsort(x)
        x = x[ind]
        y = y[..., ind]
        # y = np.take(y, ind, axis=-1)

        return PchipInterpolator(x, y, **pchipkwargs)
    else:
        return interp1d(x, y, **ipkwargs)


@public
def num_conv(v):
    """Converts a string to int, float, or str as appropriate."""
    if v is None:
        return v

    if isinstance(v, (np.str_, np.bytes_)):
        v = str(v.astype(str))

    if isinstance(v, bytes):
        return v.decode()

    for t in (int, float, str):
        try:
            if isinstance(t, int) and ("." in v or "e" in v.lower()):
                continue
            return t(v)
        except TypeError:
            continue
        except ValueError:
            continue

    raise ValueError(f"Could not convert value: {v}")


@public
def convert_temp(temp: float | str, temp_unit="C") -> float:
    """Convert temperature to Kelvin if given in Celsius or as a string with 'K' suffix."""
    if isinstance(temp, str):
        if temp == "room":
            temp = 27.0
            temp_unit = "C"
        else:
            if temp[-1].upper() in ("C", "F", "K"):
                temp_unit = temp[-1].upper()
                temp = temp[:-1].strip()
            temp = float(temp)

    if temp_unit == "F":
        temp = (temp - 32) * 5 / 9 + 273.15
        temp_unit = "C"

    if temp_unit == "C":
        temp = temp + 273.15
        temp_unit = "K"

    assert temp_unit == "K", f"Invalid temperature unit: {temp_unit}"
    return temp


@public
def dimension_round(
    min_dim: float, max_dim: float, precision: float, num_points: int
) -> np.typing.NDArray:
    dim_arr = np.logspace(
        np.log10(min_dim), np.log10(max_dim), num_points, endpoint=True
    )
    dim_arr = np.unique(
        np.round(np.asanyarray(dim_arr, dtype=float) / precision) * precision
    )
    return dim_arr[(dim_arr >= min_dim) & (dim_arr <= max_dim)]
