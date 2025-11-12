import ast
import configparser
import json
from abc import ABC, abstractmethod
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

import numpy as np

from .simulator import SIMULATORS, Simulator

LENGTH_PRECISION = 0.005  # in microns


def matrange(start, step, stop):
    num = round((stop - start) / step + 1)
    
    return np.linspace(start, stop, num)

def num_conv(v):
    for t in (int, float, str):
        try:
            return t(v)
        except ValueError:
            continue

def toupper(optionstr: str) -> str:
    return optionstr.upper()


@dataclass
class SweepConfig(ABC):
    config_file_path: str
    _configParser: configparser.ConfigParser = field(default_factory=configparser.ConfigParser, repr=False)
    _config: dict = field(init=False)
    _simulator: Simulator = field(init=False, repr=False)

    def __post_init__(self):
        self._configParser.optionxform = toupper	
        self._configParser.read(self.config_file_path)
        self._config = {s:dict(map(lambda e: (e[0], num_conv(e[1])), self._configParser.items(s))) for s in self._configParser.sections()}
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
        if self._config.get("SIMULATOR", {"TYPE": "spectre"})["TYPE"] == "spectre":
            return self._config['MODEL'].get('PARAMFILE', 'params.scs')
        else:
            return '.'.join(self._config['MODEL'].get('PARAMFILE', self._simulator.output).split(".")[:-1] + ["sch"])

    def __getitem__(self, key):
        return self._config[key]
        
    def _parse_ranges(self):
        # parse numerical ranges		
        for k in ['VGS', 'VDS', 'VSB', 'LENGTH']:
            v = ast.literal_eval(self._config['SWEEP'][k])
            v = [v] if type(v) is not list else v
            v = [matrange(*r) if isinstance(r, (list, tuple)) else [r] for r in v]
            v = list(chain.from_iterable(v))
            self._config['SWEEP'][k] = np.unique(v).tolist()

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
        kwargs.update(filter(lambda item: item[1] is not None, {'length': length, 'sb': sb}.items()))
        with open(self.paramfile, 'w') as outfile:
            outfile.write(f"parameters {' '.join([f'{k}={v}' for k, v in kwargs.items()])}")

        self._simulator.output = (length, sb)   # type: ignore
        
    def _write_netlist(self):
        """ Write the netlist for the simulation. """
        modelfile = self._config['MODEL']['FILE']
        width = self._config['SWEEP']['WIDTH']
        modelp = self._config['MODEL']['MODELP']
        modeln = self._config['MODEL']['MODELN']
        try:
            mn_supplement = '\\\n\t'.join(json.loads(self._config['MODEL']['MN']))
        except json.decoder.JSONDecodeError:
            raise SyntaxError("Error parsing config: make sure MN has no weird characters in it, and that the list isn't terminated with a trailing ','")
        try:
            mp_supplement = '\\\n\t'.join(json.loads(self._config['MODEL']['MP']))
        except json.decoder.JSONDecodeError:
            raise SyntaxError("Error parsing config: make sure MP has no weird characters in it, and that the list isn't terminated with a trailing ','")
        
        temp = float(self._config['MODEL']['TEMP']) - 273.15
        VDS_max = max(self._config['SWEEP']['VDS'])
        VDS_step = np.round(self._config['SWEEP']['VDS'][1] - self._config['SWEEP']['VDS'][0], 6)
        VGS_max = max(self._config['SWEEP']['VGS'])
        VGS_step = np.round(self._config['SWEEP']['VGS'][1] - self._config['SWEEP']['VGS'][0], 6)
        VSB_max = max(self._config['SWEEP']['VSB'])
        VSB_step = np.round(self._config['SWEEP']['VSB'][1] - self._config['SWEEP']['VSB'][0], 6)

        LEN_VEC = np.round(np.array(self._config['SWEEP']['LENGTH']) / LENGTH_PRECISION) * LENGTH_PRECISION
        NFING = self._config['SWEEP']['NFING']

        netlist = self._simulator.generate_netlist(
            modelfile=modelfile,
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
        
        n_noise.append(['mn:id', ''])
        n_noise.append(['mn:fn', ''])
        
        p_noise.append(['mp:id', ''])
        p_noise.append(['mp:fn', ''])
        return (n, p, n_noise, p_noise)
    
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
        n.append( ['n.xm1.n:ids','A',   	[1,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:vth','V',   	[0,    1,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:igd','A',   	[0,    0,   1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:igs','A',   	[0,    0,   0,    1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:gm','S',    	[0,    0,   0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:gmb','S',  	    [0,    0,   0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:gds','S',   	[0,    0,   0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:cgg','F',   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:cgdol','F',   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    1,    0,    0,    1,    0  ]])
        n.append( ['n.xm1.n:cgsol','F',   	[0,    0,   0,    0,    0,   0,    0,    1,    1,    0,    0,    0,    0,    0,    1  ]])
        n.append( ['n.xm1.n:cgs','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    1,    0,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:cgd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    1,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:cgb','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    1,    0,    0  ]])
        n.append( ['n.xm1.n:cdd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( ['n.xm1.n:cdg','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    1,    0,    0,    0  ]])
        n.append( ['n.xm1.n:css','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        n.append( ['n.xm1.n:csg','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    1,    0,    0,    0,    0,    0  ]])
        n.append( ['n.xm1.n:cjd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        n.append( ['n.xm1.n:cjs','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])

        p.append( ['n.xm2.n:ids','A',   	[1,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:vth','V',   	[0,    1,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:igd','A',   	[0,    0,   1,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:igs','A',   	[0,    0,   0,    1,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:gm','S',    	[0,    0,   0,    0,    1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:gmb','S',  	    [0,    0,   0,    0,    0,   1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:gds','S',   	[0,    0,   0,    0,    0,   0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:cgg','F',   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:cgdol','F',   	[0,    0,   0,    0,    0,   0,    0,    1,    0,    0,    1,    0,    0,    1,    0  ]])
        p.append( ['n.xm2.n:cgsol','F',   	[0,    0,   0,    0,    0,   0,    0,    1,    1,    0,    0,    0,    0,    0,    1  ]])
        p.append( ['n.xm2.n:cgs','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    1,    0,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:cgd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    1,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:cgb','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    1,    0,    0  ]])
        p.append( ['n.xm2.n:cdd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( ['n.xm2.n:cdg','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    1,    0,    0,    0  ]])
        p.append( ['n.xm2.n:css','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        p.append( ['n.xm2.n:csg','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    1,    0,    0,    0,    0,    0  ]])
        p.append( ['n.xm2.n:cjd','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ]])
        p.append( ['n.xm2.n:cjs','F',   	[0,    0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ]])
        
        n_noise.append(['n.xm1.n:id', ''])
        n_noise.append(['n.xm1.n:1overf', ''])
        
        p_noise.append(['n.xm2.n:id', ''])
        p_noise.append(['n.xm2.n:1overf', ''])
        return (n, p, n_noise, p_noise)
