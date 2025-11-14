import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator

def monotonic_interp1(x, y, **ipkwargs):
    # check for maximum monotonic subarray here
    # ...
    # placeholder
    xsub = x
    ysub = y
    # ...

    return interp1(xsub, ysub, **ipkwargs)

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

    METHOD = ipkwargs.get('kind', 'pchip')
    if METHOD == 'pchip':
        # need to convert ipkwargs
        pchipkwargs = {
            'axis'          :   ipkwargs.get('axis', 0),
            'extrapolate'   :   ipkwargs.get('extrapolate', True)
        }
        # enforce increasing monotonicity
        ind = np.argsort(x)
        x = x[ind]
        y = np.take(y, ind, axis=-1)

        return PchipInterpolator(x, y, **pchipkwargs)
    else:
        return interp1d(x, y, **ipkwargs)

def num_conv(v):
    """ Converts a string to int, float, or str as appropriate. """
    if v is None:
        return v
    
    if isinstance(v, (np.str_, np.bytes_)):
        v = str(v.astype(str))
    
    if isinstance(v, bytes):
        return v.decode()
    
    for t in (int, float, str):
        try:
            if t == int and ('.' in v or 'e' in v.lower()):
                continue
            return t(v)
        except TypeError:
            continue
        except ValueError:
            continue

    raise ValueError(f"Could not convert value: {v}")


def convert_temp(temp: float | str, temp_unit="C") -> float:
    """ Convert temperature to Kelvin if given in Celsius or as a string with 'K' suffix. """
    if isinstance(temp, str):
        if temp == "room":
            temp = 27.0
            temp_unit = "C"
        else:
            if temp[-1].upper() in ("C", "F", "K"):
                temp_unit = temp[-1].upper()
                temp = temp[:-1]
            temp = float(temp)

    return temp + (273.15 if temp_unit == "C" else 0)
