import glob
import os
import pickle
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from functools import cached_property, partial, wraps
from itertools import chain
from pathlib import Path
from typing import *  # type: ignore

import h5py
import numpy as np
import prettytable
import scipy.io
from auto_all import public
from scipy.interpolate import interpn

from .constants import *
from .numerical import interp1, convert_temp


@dataclass
class _BaseLUT(Mapping):
    """Base LUT implementing mapping protocol for Lookup class."""
    filename: str = ""
    device: InitVar[Optional[str]] = None
    lut_kwargs: InitVar[Dict] = field(default={})

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
        return len(self.keys())

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable entries
        if '_h5file' in state:
            del state['_h5file']
        return state

    def __str__(self) -> str:
        return f"filename={self.filename}"


@dataclass
class _PKLLUT(_BaseLUT):
    def __post_init__(self, lut_kwargs):
        with open(self.filename, 'rb') as f:
            data = pickle.load(f)
        # normalize keys to upper
        self.data = {k.upper(): v for k, v in data.items()}
        for k, v in lut_kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        k = key.upper()
        val = self.data[k]
        return deepcopy(val) if not isinstance(val, dict) else {kk: deepcopy(vv) for kk, vv in val.items()}


@dataclass
class _MATLUT(_BaseLUT):
    def __post_init__(self, lut_kwargs):
        mat = scipy.io.loadmat(self.filename, matlab_compatible=True)
        # find first non-header key
        for k in mat.keys():
            if not( k.startswith('__') and k.endswith('__') ):
                mat_struct = mat[k]
                break
        else:
            raise RuntimeError('No valid data found in .mat file')

        # MATLAB struct array nesting: take first element
        # mat_struct is a numpy structured array
        self.data = {k.upper(): deepcopy(np.squeeze(mat_struct[k][0][0])) for k in mat_struct.dtype.names}
        for k, v in lut_kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        k = key.upper()
        val = self.data[k]
        return deepcopy(val) if not isinstance(val, dict) else {kk: deepcopy(vv) for kk, vv in val.items()}


def h5open(func: [Callable]=None, *, cls_override: Optional[Any]=None):
    if func is None:
        return partial(h5open, cls_override=cls_override)

    @wraps(func)
    def open_h5(cls, *args, **kwargs):
        c = cls_override or cls
        # consider the file "not opened" unless it's an h5py.File instance
        if (closing := not isinstance(c._h5file, h5py.File)):
            c._h5file = h5py.File(c.filename, 'r')

        assert isinstance(c._h5file, h5py.File), "HDF5 file not opened"
        res = func(cls, *args, **kwargs)
        
        if closing:
            c._h5file.close()
            c._h5file = None
        return res
    return open_h5

@dataclass
class _H5LUT(_BaseLUT):
    device: Optional[str] = None
    _h5file: Optional[h5py.File] = field(default=None, repr=False)

    def __post_init__(self, lut_kwargs):
        self.env_kwargs = {k.upper(): v for k, v in lut_kwargs.items()}

    @property
    def env_kwargs(self):
        if not hasattr(self, '_env_kwargs'):
            self.env_kwargs = {}
        return self._env_kwargs
    
    @env_kwargs.setter
    def env_kwargs(self, val: Dict):
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
        # traverse environment keys that are of the form KEY:val
        while len(env_keys := set(map(lambda k: k.split(":")[0], grp.keys()))) == 1:  # type: ignore
            k = next(iter(env_keys))
            num_conv = lambda e: convert_temp(e, temp_unit='K') if k == 'TEMP' else unit_init(e)
            grp_keys = list(k.split(":")[1] for k in grp.keys())     # type: ignore

            env_val = to_magnitude(num_conv(self.env_kwargs.get(k.upper(), globals().get(k.upper(), os.getenv(k.upper())))))
            assert env_val is not None, f"Environment variable {k} not specified!"

            if isinstance(env_val, str):
                chosen = env_val
            else:
                dist_calc = lambda x: abs(x - env_val) if x <= env_val or k != 'VDD' or np.isclose(x, env_val) else float('inf')    # type: ignore
                chosen = grp_keys[np.argmin([dist_calc(to_magnitude(unit_init(ck))) for ck in grp_keys])]   # type: ignore

            grp = grp[f"{k}:{chosen}"]

        # Load by device
        if self.device is None and set(grp.keys()) == {'n', 'p'}:   # type: ignore
            raise ValueError("Device type must be specified when both n and p data are present in the file.")
        elif self.device is not None:
            grp = grp[self.device]

        return grp.name # type: ignore

    @h5open
    def keys(self) -> KeysView[Any]:
        return KeysView(list(self._h5file[self.lut_key].keys()))    # type: ignore
    
    @h5open
    def values(self) -> ValuesView[Any]:
        return ValuesView(list(self._h5file[self.lut_key].values()))    # type: ignore

    @h5open
    def items(self) -> ItemsView[Any, Any]:
        return ItemsView(list(self._h5file[self.lut_key].items()))  # type: ignore

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


@public
@dataclass
class Lookup:
    filename: InitVar[Optional[str]] = None
    device: InitVar[Optional[str]] = None
    lut_kwargs: InitVar[Dict] = field(default={})
    _mode: int = field(init=False, default=1, repr=False)

    @property
    def __DATA(self):
        if not hasattr(self, "data"):
            self.data = {}
        return self.data

    @__DATA.setter
    def __DATA(self, val: Dict):
        if (filename := val.get('filename', None)) is not None:
            # Choose appropriate LUT subclass based on file extension
            LUTS = {
                '.mat': _MATLUT,
                '.pkl': _PKLLUT,
                '.h5' : _H5LUT,
                '.hdf5': _H5LUT,
            }
            val = LUTS[Path(filename).suffix](**val)
        self.data = val

    @property
    def __modefuncmap(self) -> Callable:
        f = {   
            1 : self._SimpleLK,
            2 : self._SimpleLK,  
            3 : self._RatioVRatioLK
        }[self._mode]
        if isinstance(self.__DATA, _H5LUT):
            f = h5open(f, cls_override=self.__DATA)   # type: ignore
        return f

    @__modefuncmap.setter
    def __modefuncmap(self, args: Tuple):
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
        elif out_ratio and (not var_ratio):
            self._mode = 2
        elif (not out_ratio) and (not var_ratio):
            self._mode = 1
        else:
            raise ValueError("Invalid syntax or usage mode! Please check documentation.")

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
        kwargs = {k.upper(): v for k, v in lut_kwargs.items()} # convert kwargs to upper
        self.__load(filename, device, **kwargs)
        self.__default = {
            'L'     :   kwargs.get('L', min(self.__DATA['L'])),
            'VGS'   :   kwargs.get('VGS', self.__DATA['VGS']),
            'VDS'   :   kwargs.get('VDS', max(self.__DATA['VDS'])/2),
            'VSB'   :   kwargs.get('VSB', 0.0),
            'METHOD':   kwargs.get('METHOD', 'pchip'),
            'VGB'   :   kwargs.get('VGB', None),
            'GM_ID' :   kwargs.get('GM_ID', None),
            'ID_W'  :   kwargs.get('ID_W', None),
            'VDB'   :   kwargs.get('VDB', None)
        }

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
            techsweep_dir = os.getenv("TECHSWEEP_DIR", os.path.expandvars("$PDK_ROOT/techsweeps"))
            filename = os.path.join(techsweep_dir, next(chain.from_iterable(map(lambda ext: glob.iglob(f'*{ext}', root_dir=techsweep_dir), ['.h5', '.hdf5', '.mat', '.pkl']))))

        try:
            self.__DATA = dict(filename=filename, device=device, lut_kwargs=kwargs)
        except KeyError:
            raise TypeError(f'File not supported (only .mat, .pkl, .h5 and .hdf5): {filename}')


    def __contains__(self, key):
        return key.upper() in self.__DATA.keys() or any(isinstance(v, dict) and key.upper() in v for v in self.__DATA.values())

    def __getitem__(self, key):
        """
        __getitem__ dunder method overwritten to allow convenient
        pseudo array access to member data. Returns a copy of the
        member array.
        """
        if key not in self:
            raise ValueError(f"Lookup table does not contain this data")

        if key.upper() in self.__DATA:
            return np.copy(self.__DATA[key.upper()])
        else:
            k = next(filter(lambda x: isinstance(self.__DATA[x], dict) and key.upper() in self.__DATA[x], self.__DATA.keys()))
            return np.copy(self.__DATA[k][key]) 

    def __setitem__(self, key, value):
        """
        __setitem__ dunder method overwritten to allow convenient
        pseudo array access to member data. Sets the member data
        to the value passed.
        """
        if key not in self:
            raise ValueError(f"Lookup table does not contain this data")
        
        if key.upper() in self.__DATA:
            self.__DATA[key.upper()] = np.copy(value)
        else:
            k = next(filter(lambda x: isinstance(self.__DATA[x], dict) and key.upper() in self.__DATA[x], self.__DATA.keys()))
            self.__DATA[k][key] = np.copy(value) 

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
        outkeys = out.upper().split('_')
        varkeys, vararg = next(iter((kwargs.items()))) if kwargs else (None, None)
        varkeys = str(varkeys).upper().split('_')

        kwargs = {k.upper(): v for k, v in kwargs.items()} # convert kwargs to upper
        defaultdict = {k:self.__default.get(k) for k in ['L', 'VGS', 'VDS', 'VSB', 'METHOD']}
        pars = {k:kwargs.get(k, v) for k, v in defaultdict.items()} # extracts parameters from kwargs
        
        # common kwargs for interpolating functions
        ipkwargs = {'bounds_error': False,
                    'fill_value' : None}

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
            with np.errstate(divide='ignore',invalid='ignore'):
                ydata =  self.__DATA[num]/self.__DATA[den]
                # nan causing issues with interpn extrapolation
                ydata[np.isnan(ydata)] *= 0.0
        else:
            outkey = outkeys[0]
            ydata = self.__DATA[outkey]

        points = (self.__DATA['L'], self.__DATA['VGS'], self.__DATA['VDS'],\
            self.__DATA['VSB'])
        xi_mesh = np.array(np.meshgrid(pars['L'], pars['VGS'], pars['VDS'], pars['VSB'], indexing='ij'))
        xi = np.rollaxis(xi_mesh, 0, 5)
        xi = xi.reshape(int(xi_mesh.size/4), 4)

        output = interpn(points, ydata, xi, **ipkwargs).reshape(len(np.atleast_1d(pars['L'])), \
            len(np.atleast_1d(pars['VGS'])), len(np.atleast_1d(pars['VDS'])),\
                 len(np.atleast_1d(pars['VSB'])) )
        
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
        with np.errstate(divide='ignore',invalid='ignore'):    
            # unpack outkeys and ydata
            num, den = outkeys
            ydata =  self.__DATA[num]/self.__DATA[den]
            ydata[np.isnan(ydata)] *= 0.0
            # unpack varkeys and xdata
            num, den = varkeys
            xdata = self.__DATA[num]/self.__DATA[den]
            xdata[np.isnan(xdata)] *= 0.0

        xdesired = np.atleast_1d(vararg)
        
        points = (self.__DATA['L'], self.__DATA['VGS'], self.__DATA['VDS'],\
            self.__DATA['VSB'])
        xi_mesh = np.array(np.meshgrid(pars['L'], pars['VGS'], pars['VDS'], pars['VSB'], indexing='ij'))
        xi = np.rollaxis(xi_mesh, 0, 5)
        xi = xi.reshape(int(xi_mesh.size/4), 4)

        x = interpn(points, xdata, xi, **ipkwargs).reshape(len(np.atleast_1d(pars['L'])), \
            len(np.atleast_1d(pars['VGS'])), len(np.atleast_1d(pars['VDS'])),\
                 len(np.atleast_1d(pars['VSB'])))
        
        y = interpn(points, ydata, xi, **ipkwargs).reshape(len(np.atleast_1d(pars['L'])), \
            len(np.atleast_1d(pars['VGS'])), len(np.atleast_1d(pars['VDS'])),\
                 len(np.atleast_1d(pars['VSB'])))
        
        x = np.array(np.squeeze(np.transpose(x, (1, 0, 2, 3))))
        y = np.array(np.squeeze(np.transpose(y, (1, 0, 2, 3))))
        
        if x.ndim == 1:
            x.shape += (1,)
            y.shape += (1,)

        dim = x.shape
        output = np.zeros((dim[1], len(xdesired)))  #   type: ignore
        ipkwargs = {
            'kind' : pars['METHOD'],
            'fill_value' : np.nan
        }
        
        for i in range(0, dim[1]):
            for j in range(0, len(xdesired)):
                m = max(x[:, i])
                idx = np.argmax(x[:, i])
                if (xdesired[j] > m):
                    print(f'Look up warning: {num}_{den} input larger than maximum! Output is NaN')
                if (num.upper() == 'GM') and (den.upper() == 'ID'):
                    x_right = x[idx:-1, i]
                    y_right = y[idx:-1, i]
                    output[i, j] = interp1(x_right, y_right, **ipkwargs)(xdesired[j])
                elif (num.upper() == 'GM') and (den.upper() == 'CGG') or (den.upper() == 'CGG'):
                    x_left = x[:idx, i]
                    y_left = y[:idx, i]
                    output[i, j] = interp1(x_left, y_left, **ipkwargs)(xdesired[j])
                else:
                    crossings = len(np.argwhere(np.diff(np.sign(x[:,i]-xdesired[j]+eps))))
                    if crossings > 1:
                        print('Crossing warning')
                        return []
                    output[i, j] = interp1(x[:,i], y[:, i], **ipkwargs)(xdesired[j])

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
            kwargs = {k.upper(): v for k, v in kwargs.items()} # convert kwargs to upper
            defaultdict = {k:self.__default.get(k) for k in ['L', 'VDS', 'VDB', 'VGB', 'GM_ID', 'ID_W', 'VSB', 'METHOD']}
            pars = {k:kwargs.get(k, v) for k,v in defaultdict.items()}

            #Check whether GM_ID or ID_W was passed to function
            ratio_string = 'None'
            ratio_data = None

            if pars['ID_W'] is not None:
                ratio_string = 'ID_W'
                ratio_data = pars['ID_W']

            elif pars['GM_ID'] is not None:
                ratio_string = 'GM_ID'
                ratio_data = pars['GM_ID']
            
            # determining the mode 
            # In usage mode (1), the inputs to the function are GM_ID (or ID/W), L, 
            # VDS and VSB
            if (pars['VGB'] and pars['VDB']) == None:
                mode = 1
            # In usage mode (2), VDB and VGB must be supplied to the function
            elif (pars['VGB'] and pars['VDB']) != None:
                mode = 2
            else:
                raise SyntaxError("Invalid syntax or usage mode!")
            
            if mode == 1:
                VGS = self.__DATA['VGS']
                ratio = to_magnitude(self.look_up(ratio_string, VGS = VGS, VDS=pars['VDS'], VSB=pars['VSB'], L=pars['L']))
            elif mode == 2:
                step = self.__DATA['VGS'][0] - self.__DATA['VGS'][1]
                VSB = np.arange(max(self.__DATA['VSB']), min(self.__DATA['VSB']) + step, step)
                VGS = pars['VGB'] - VSB
                VDS = pars['VDB'] - VSB
                ratio = np.array([to_magnitude(self.look_up(ratio_string, VGS=VGS[i], VDS=VDS[i], VSB=VSB[i], L=pars['L'])).item() for i in range(len(VGS))])
                idx = ~np.isnan(ratio)
                ratio = ratio[idx]
                VGS = VGS[idx]
            else:
                raise RuntimeError("Invalid mode selected!")
    
            if (np.size(pars['L']) == 1):
                ratio.shape += (1,)
            else:
                ratio = np.swapaxes(ratio, 0, 1)

            s = ratio.shape
            
            output = np.empty((s[1], len(np.atleast_1d(ratio_data))))   # type: ignore
            output[:] = np.nan
            
            m = np.max(ratio)
            for j in range(s[1]):
                ratio_range = ratio[:,j]
                VGS_range = VGS

                if ratio_string == 'GM_ID':
                    idx = np.where(ratio == m)[0].item()
                    VGS_range = VGS_range[idx:]
                    ratio_range = ratio_range[idx:]

                    if np.max(np.atleast_1d(ratio_data)) > m:  # type: ignore
                        print('look_upVGS: GM_ID input larger than maximum!')
                
                output[j,:] = interp1(ratio_range, VGS_range)(ratio_data)
                output = output[:]
            
            return np.squeeze(output)   # type: ignore

        if isinstance(self.__DATA, _H5LUT):
            perform_lk = h5open(perform_lk, cls_override=self.__DATA)   # type: ignore
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
        return self.look_up('STH_GM', **kwargs)/(4*kB*self['TEMP'].item())
    
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
        return self.look_up('SFL_STH', **kwargs)
        
    def __repr__(self) -> str:
        return f"PyGMID_Lookup<{self.__DATA['INFO']}>"

    def __str__(self):
        tab = prettytable.PrettyTable()
        tab.title = f"PyGMID: {self.__DATA['INFO']}"
        tab.field_names = ['Variable', 'Size', 'Min', 'Max']

        for k, v in self.__DATA.items():
            if not hasattr(v, 'dtype'):
                continue
            is_numeric = np.issubdtype(v.dtype, np.number)

            size = None

            if is_numeric :
                size = str(v.shape).replace('(', '').replace(')', '').\
                                    replace(', ', 'x').replace(',', '')

            tab.add_row([ k
                        , size             if size       else '1'
                        , f'{v.min():.2e}' if is_numeric else 'N/A'
                        , f'{v.max():.2e}' if is_numeric else 'N/A'])

        return f"PyGMID (from {self.__DATA}):\n{tab}"
