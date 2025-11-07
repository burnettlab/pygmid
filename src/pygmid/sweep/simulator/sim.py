""" Simulator base class and utilities for sweep simulations. """
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import os

import numpy as np


def multiline_join(in_str: str) -> str:
    ix, line = next(filter(lambda l: len(l[1].lstrip()) and l[1].lstrip()[0] != l[1][0], enumerate(in_str.splitlines())))
    indent_amt = len(line) - len(line.lstrip())
    return '\n'.join(
        map(lambda e: e[1][(0 if e[0] < ix else indent_amt):], enumerate(in_str.splitlines()))
    )



@dataclass
class Simulator(ABC):
    """ Abstract base class for sweep simulators. """
    _config: 'Config' = field(repr=False)   # type: ignore
    netlist_name: str = 'pysweep'
    netlist_ext: str = field(init=False)
    args: List[str] = field(default_factory=lambda: [os.getcwd()])
    _sweep_dir: str = './sweep'

    def __post_init__(self):
        pass

    @property
    def directory(self) -> str:
        return os.path.expandvars(self.args[-1])

    @directory.setter
    def directory(self, dir: str):
        self.args[-1] = dir

    @property
    def netlist_filepath(self) -> str:
        return os.path.expandvars(f"{self.netlist_name}.{self.netlist_ext}")
    
    @property
    @abstractmethod
    def output(self) -> str:
        pass
    
    @output.setter
    @abstractmethod
    def output_setter(self, args: Tuple):
        pass

    @abstractmethod
    def generate_netlist(self, **kwargs) -> str:
        pass

    @abstractmethod
    def run(self) -> Tuple[str, str]:
        pass

    @abstractmethod
    def _run_sim(self):
        pass

    @abstractmethod
    def extract_sweep_params(self, sweep_output_directory, sweep_type) -> Tuple[Dict, Dict]:
        pass
            
    def parse_sim(self, filepath):
        fileparts = filepath.split("_")
        i = int(fileparts[-2])
        j = int(fileparts[-1])
        
        (n_dict, p_dict) = self.extract_sweep_params(filepath, sweep_type="DC")
        (nn_dict, pn_dict) = self.extract_sweep_params(filepath, sweep_type="NOISE")

        return i, j, n_dict, p_dict, nn_dict, pn_dict

    def _cleanup(self, nch, pch) -> Tuple[str, str]:
        try:
            if os.path.exists(self._sweep_dir):
                shutil.rmtree(self._sweep_dir)
            os.remove(self.netlist_filepath)
        except OSError as e:
            print(f"Could not perform cleanup:\nFile - {e.filename}\nError - {e.strerror}")

        # then save data to file
        modeln_file_path = f"{self._config['MODEL']['SAVEFILEN']}.pkl"
        modelp_file_path = f"{self._config['MODEL']['SAVEFILEP']}.pkl"
        with open(modeln_file_path, 'wb') as f:
            pickle.dump(nch, f)
        with open(modelp_file_path, 'wb') as f:
            pickle.dump(pch, f)
        return (modeln_file_path, modelp_file_path)