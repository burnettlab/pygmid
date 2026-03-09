import logging
import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Dict
from warnings import warn

from auto_all import public

from .config import NGSpiceConfig, SpectreConfig, SweepConfig
from .simulator import *


LOGGER = logging.getLogger(__name__)


@public
@dataclass
class Sweep:
    config_file_path: str
    _config: SweepConfig = field(init=False, repr=False)
    
    def __post_init__(self):
        cfg_dir = os.path.dirname(os.path.abspath(self.config_file_path))
        LOGGER.info(f"Searching for config in directory: {cfg_dir}")
        for f in filter(lambda p: p.suffix == ".py", map(lambda p: Path(p), os.listdir(cfg_dir))):
            # Import the file and check if it has a class that is a subclass of Config
            if (rel_path := os.path.relpath(cfg_dir, os.getcwd())) != ".":
                module_name = f"{rel_path.replace(os.sep, '.')}.{f.stem}"
            else:
                module_name = f.stem
            LOGGER.debug(f"Trying to load module: {module_name}")
            module = import_module(module_name)
            try:
                cls = next(filter(lambda c: isinstance(c, type) and issubclass(c, SweepConfig) and c != SweepConfig, map(lambda n: getattr(module, n), filter(lambda n: not n.startswith("__") and not n.endswith("__"), dir(module)))))
                self._config = cls(self.config_file_path)
                LOGGER.debug(f"Loaded config from {f.stem}{f.suffix}")
                break
            except StopIteration:
                pass

        if getattr(self, '_config', None) is None:
            warn("No Config subclass found in the current directory. Using default Config class.", ImportWarning)
            configs: Dict[str, SweepConfig] = {
                'ngspice': NGSpiceConfig,
                'spectre': SpectreConfig,
            }   # type: ignore
            for sim_name, config in configs.items():
                cfg = config(self.config_file_path)     # type: ignore
                if cfg._config['SIMULATOR']["TYPE"].lower() == sim_name:
                    self._config = cfg
                    LOGGER.debug(f"Loaded {sim_name} config from default Config class.")
                    break
        
        self._config._write_netlist()

    @property
    def _simulator(self) -> 'Simulator':
        return self._config._simulator
    
    def run(self):
        return self._simulator.run()
