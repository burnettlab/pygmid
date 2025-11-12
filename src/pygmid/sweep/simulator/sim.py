""" Simulator base class and utilities for sweep simulations. """
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import os
import scipy.io
import h5py

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
        model_paths = []

        for savefile, data in zip([self._config['MODEL']['SAVEFILEN'], self._config['MODEL']['SAVEFILEP']], [nch, pch]):
            file_root, file_ext = os.path.splitext(savefile)
            if not file_ext:
                file_ext = '.pkl'

            filename = f"{file_root}{file_ext}"
            if file_ext == '.mat':
                scipy.io.savemat(filename, data)
            elif file_ext == '.pkl':
                with open(filename, 'rb') as f:
                    pickle.dump(data, f)
            elif file_ext == '.hdf5':
                with h5py.File(filename, 'w') as f:
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
            else:
                raise TypeError(f'Filetype {file_ext} not supported (only .mat, .pkl and .hdf5)')
            
            model_paths.append(filename)

        return tuple(model_paths)