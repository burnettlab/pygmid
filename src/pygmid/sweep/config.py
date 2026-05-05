import ast
import configparser
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
from auto_all import public

from ..numerical import convert_temp, num_conv
from .simulator import *

LENGTH_PRECISION = 0.005  # in microns


def matrange(start, step, stop):
    num = round((stop - start) / step + 1)
    
    return np.linspace(start, stop, num)

def toupper(optionstr: str) -> str:
    return optionstr.upper()


@public
@dataclass
class SweepConfig(ABC):
    config_file_path: str
    _configParser: configparser.ConfigParser = field(default_factory=configparser.ConfigParser, repr=False)
    _config: dict = field(init=False)
    _simulator: 'Simulator' = field(init=False, repr=False)

    def __post_init__(self):
        def convert_string_paths(d: dict) -> dict:
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = convert_string_paths(v)
                elif isinstance(v, str):
                    d[k] = ' '.join(map(os.path.expandvars, v.split(' ')))
            return d
        
        self._configParser.optionxform = toupper	
        self._configParser.read(self.config_file_path)
        self._config = convert_string_paths({s:dict(map(lambda e: (e[0], num_conv(e[1])), self._configParser.items(s))) for s in self._configParser.sections()})
        self._parse_ranges()
        
        self._config['outvars'] = 	['ID','VT','IGD','IGS','GM','GMB','GDS','CGG','CGS','CSG','CGD','CDG','CGB','CDD','CSS']
        self._config['outvars_noise'] = ['STH','SFL']
        n, p, n_noise, p_noise = self.generate_outvars()
        self._config['n'] = n
        self._config['p'] = p
        self._config['n_noise'] = n_noise
        self._config['p_noise'] = p_noise

        self._simulator = SIMULATORS[self._config.get("SIMULATOR", {"TYPE": "spectre"})["TYPE"]](self)

    @property
    def paramfile(self) -> str:
        return os.path.expandvars(self._config['MODEL'].get('PARAMFILE', 'params.scs') if self._config.get("SIMULATOR", {"TYPE": "spectre"})["TYPE"] == "spectre" else '.'.join(self._config['MODEL'].get('PARAMFILE', self._simulator.output).split(".")[:-1] + ["sch"]))

    def __getitem__(self, key):
        return self._config[key]
        
    def _parse_ranges(self):
        # parse numerical ranges		
        for k in ['VGS', 'VDS', 'VSB', 'LENGTH']:
            v = ast.literal_eval(self._config['SWEEP'][k])
            v = [v] if type(v) is not list else v
            v = [matrange(*r) if isinstance(r, (list, tuple)) else [r] for r in v]
            v = list(chain.from_iterable(v))
            self._config['SWEEP'][k] = np.unique(v).squeeze()

        self._config['SWEEP']['WIDTH'] = float(self._config['SWEEP']['WIDTH'])
        self._config['SWEEP']['NFING'] = int(self._config['SWEEP']['NFING'])
    
    def generate_m_dict(self):
        m_dict = self._config.get('SPEC', {})
        m_dict.update({
            'INFO' : self._config['MODEL']['INFO'],
            'CORNER' : self._config['MODEL']['CORNER'],
            'TEMP' : self._config['MODEL']['TEMP'],
            'NFING' : self._config['SWEEP']['NFING'],
            'L' : np.array(self._config['SWEEP']['LENGTH']).T,
            'W' : self._config['SWEEP']['WIDTH'],
            'VGS' : np.array(self._config['SWEEP']['VGS']).T,
            'VDS' : np.array(self._config['SWEEP']['VDS']).T,
            'VSB' : np.array(self._config['SWEEP']['VSB']).T 
        })
        return m_dict.copy()

    @abstractmethod
    def write_params(self, length: Optional[Union[float, str]] = None, sb: Optional[Union[float, str]] = None, **kwargs):
        with open(self.paramfile, 'w') as outfile:
            outfile.write(f"parameters length={length} sb={sb}\n")

        self._simulator.output = kwargs.get('index', (length, sb))
        
    def _write_netlist(self):
        """ Write the netlist for the simulation. """
        model_keys = self._config['MODEL'].keys()
        width = self._config['SWEEP']['WIDTH']
        modelp = self._config['MODEL']['MODELP']
        modeln = self._config['MODEL']['MODELN']

        try:
            mn_supplement = ' \\\n\t'.join(map(os.path.expandvars, json.loads(self._config['MODEL']['MN'])))
        except json.decoder.JSONDecodeError:
            raise SyntaxError("Error parsing config: make sure MN has no weird characters in it, and that the list isn't terminated with a trailing ','")
        try:
            mp_supplement = ' \\\n\t'.join(map(os.path.expandvars, json.loads(self._config['MODEL']['MP'])))
        except json.decoder.JSONDecodeError:
            raise SyntaxError("Error parsing config: make sure MP has no weird characters in it, and that the list isn't terminated with a trailing ','")

        div_arr = lambda n, d: np.isclose(n / d, np.array([np.floor(n / d), np.ceil(n / d)]))
        is_divisible = lambda n, d: np.any(div_arr(n, d))

        temp = convert_temp(self._config['MODEL']['TEMP'])
        VDS_max = max(self._config['SWEEP']['VDS'])
        VDS_step = np.round(self._config['SWEEP']['VDS'][1] - self._config['SWEEP']['VDS'][0], 6)
        assert is_divisible(VDS_max, VDS_step), f"VDS Maximum ({VDS_max}) must be divisible by step size ({VDS_step}) ({div_arr(VDS_max, VDS_step)})"
        VGS_max = max(self._config['SWEEP']['VGS'])
        VGS_step = np.round(self._config['SWEEP']['VGS'][1] - self._config['SWEEP']['VGS'][0], 6)
        assert is_divisible(VGS_max, VGS_step), f"VGS Maximum ({VGS_max}) must be divisible by step size ({VGS_step}) ({div_arr(VGS_max, VGS_step)})"
        VSB_max = max(self._config['SWEEP']['VSB'])
        VSB_step = np.round(self._config['SWEEP']['VSB'][1] - self._config['SWEEP']['VSB'][0], 6)
        assert is_divisible(VSB_max, VSB_step), f"VSB Maximum ({VSB_max}) must be divisible by step size ({VSB_step})"

        LEN_VEC = np.round(np.array(self._config['SWEEP']['LENGTH']) / LENGTH_PRECISION) * LENGTH_PRECISION
        NFING = self._config['SWEEP']['NFING']

        # Remove MODEL keys passed in other ways
        model_keys -= {'MODELP', 'MODELN', 'MN', 'MP', 'TEMP'}

        netlist = self._simulator.generate_netlist(
            paramfile=self.paramfile,
            width=width,
            modelp=modelp,
            modeln=modeln,
            mn_supplement=mn_supplement,
            mp_supplement=mp_supplement,
            temp=temp,
            VDS_max=VDS_max,
            VDS_step=VDS_step,
            VGS_max=VGS_max,
            VGS_step=VGS_step,
            VSB_max=VSB_max,
            VSB_step=VSB_step,
            LEN_VEC=LEN_VEC,
            NFING=NFING,
            **dict(map(lambda k: (f"model{k.lower()}", self._config['MODEL'][k]), model_keys))
        )        

        with open(self._simulator.netlist_filepath, 'w') as f:
            f.write(netlist)
    
    @abstractmethod
    def generate_outvars(self, n: List=[], p: List=[], n_noise: List=[], p_noise: List=[]) -> Tuple[List, List, List, List]:
        """ Generate the mapping of output variables from the simulation to the lookup table. 
        
        outvars: `['ID','VT','IGD','IGS','GM','GMB','GDS','CGG','CGS','CSG','CGD','CDG','CGB','CDD','CSS']`
        outvars_noise: `['STH','SFL']`

        """
        pass


@public
class SpectreConfig(SweepConfig):
    """ Configuration class for sweep simulations using Spectre. """
    def __post_init__(self):
        super().__post_init__()
    
    def write_params(self, length: Optional[Union[float, str]]=None, sb: Optional[Union[float, str]]=None, **kwargs):
        return super().write_params(length, sb, **kwargs)
    
    def generate_outvars(self, n: List=[], p: List=[], n_noise: List=[], p_noise: List=[]) -> Tuple[List, List, List, List]:
        """ Generate the mapping of output variables from the simulation to the lookup table. 
        
        outvars: `['ID','VT','IGD','IGS','GM','GMB','GDS','CGG','CGS','CSG','CGD','CDG','CGB','CDD','CSS']`
        outvars_noise: `['STH','SFL']`

        """
        n.append( ['mn:ids','A',   	[1,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:vth','V',   	[0,    1,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:igd','A',   	[0,    0,   1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:igs','A',   	[0,    0,   0,    1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:gm','S',    	[0,    0,   0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:gmbs','S',  	[0,    0,   0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:gds','S',   	[0,    0,   0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cgg','F',   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cgs','F',   	[0,    0,   0,    0,    0,   0,    0,    0,   -1,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cgd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,   -1,    0,    0,    0,    0  ]])
        n.append( ['mn:cgb','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,   -1,    0,    0  ]])
        n.append( ['mn:cdd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( ['mn:cdg','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,   -1,    0,    0,    0  ]])
        n.append( ['mn:css','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        n.append( ['mn:csg','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,   -1,    0,    0,    0,    0,    0  ]])
        n.append( ['mn:cjd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( ['mn:cjs','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])

        p.append( ['mp:ids','A',   	[-1,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:vth','V',   	[ 0,   -1,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:igd','A',   	[ 0,    0,   -1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:igs','A',   	[ 0,    0,    0,   -1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:gm','S',    	[ 0,    0,    0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:gmbs','S',  	[ 0,    0,    0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:gds','S',   	[ 0,    0,    0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cgg','F',   	[ 0,    0,    0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cgs','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,   -1,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cgd','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,   -1,    0,    0,    0,    0  ]])
        p.append( ['mp:cgb','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,   -1,    0,    0  ]])
        p.append( ['mp:cdd','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( ['mp:cdg','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,   -1,    0,    0,    0  ]])
        p.append( ['mp:css','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        p.append( ['mp:csg','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,   -1,    0,    0,    0,    0,    0  ]])
        p.append( ['mp:cjd','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( ['mp:cjs','F',   	[ 0,    0,    0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        
        n_noise.append(['mn:id', '', 1])
        n_noise.append(['mn:fn', '', 1])
        
        p_noise.append(['mp:id', '', 1])
        p_noise.append(['mp:fn', '', 1])
        return (n, p, n_noise, p_noise)
    

@public
class NGSpiceConfig(SweepConfig):
    """ Configuration class for sweep simulations using ngspice. """
    def __post_init__(self):
        super().__post_init__()
    
    def write_params(self, length: Optional[Union[float, str]]=None, sb: Optional[Union[float, str]]=None, **kwargs):
        return super().write_params(length, sb, **kwargs)
    
    def generate_outvars(self, n: List=[], p: List=[], n_noise: List=[], p_noise: List=[]) -> Tuple[List, List, List, List]:
        """ Generate the mapping of output variables from the simulation to the lookup table. 
        
        outvars: `['ID','VT','IGD','IGS','GM','GMB','GDS','CGG','CGS','CSG','CGD','CDG','CGB','CDD','CSS']`
        outvars_noise: `['STH','SFL']`

        """
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[ids]","A",   	[1,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[vth]","V",   	[0,    1,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[igd]","A",   	[0,    0,   1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[igs]","A",   	[0,    0,   0,    1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[gm]","S",    	[0,    0,   0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[gmb]","S",  	    [0,    0,   0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[gds]","S",   	[0,    0,   0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cgg]","F",   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cgdol]","F",   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    1,    0,    0,    1,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cgsol]","F",   	[0,    0,   0,    0,    0,   0,    0,    1,    1,    1,    0,    0,    0,    0,    1  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cgs]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    1,    1,    0,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cgd]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    1,    0,    0,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cgb]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    1,    0,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cdd]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[css]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cjd]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( [f"@n.xm1.n{self._config['MODEL']['MODELN']}[cjs]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])

        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[ids]","A",   	[1,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[vth]","V",   	[0,    1,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[igd]","A",   	[0,    0,   1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[igs]","A",   	[0,    0,   0,    1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[gm]","S",    	[0,    0,   0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[gmb]","S",  	    [0,    0,   0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[gds]","S",   	[0,    0,   0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cgg]","F",   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cgdol]","F",   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    1,    0,    0,    1,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cgsol]","F",   	[0,    0,   0,    0,    0,   0,    0,    1,    1,    1,    0,    0,    0,    0,    1  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cgs]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    1,    1,    0,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cgd]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    1,    0,    0,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cgb]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    1,    0,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cdd]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[css]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cjd]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( [f"@n.xm2.n{self._config['MODEL']['MODELP']}[cjs]","F",   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        
        n_noise.append([f"@n.xm1.n{self._config['MODEL']['MODELN']}[sid]", "", 1])
        n_noise.append([f"@n.xm1.n{self._config['MODEL']['MODELN']}[sfl]", "", 1])
        
        p_noise.append([f"@n.xm2.n{self._config['MODEL']['MODELP']}[sid]", "", 1])
        p_noise.append([f"@n.xm2.n{self._config['MODEL']['MODELP']}[sfl]", "", 1])
        return (n, p, n_noise, p_noise)
