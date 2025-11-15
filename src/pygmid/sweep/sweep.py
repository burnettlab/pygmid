import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from warnings import warn
from typing import Dict

from .config import SweepConfig, SpectreConfig, NGSpiceConfig
from .simulator import Simulator


@dataclass
class Sweep:
    config_file_path: str
    _config: SweepConfig = field(init=False, repr=False)
    
    def __post_init__(self):
        cfg_dir = os.path.dirname(os.path.abspath(self.config_file_path))
        print(f"Searching for config in directory: {cfg_dir}")
        for f in filter(lambda p: p.suffix == ".py", map(lambda p: Path(p), os.listdir(cfg_dir))):
            # Import the file and check if it has a class that is a subclass of Config
            if (rel_path := os.path.relpath(cfg_dir, os.getcwd())) != ".":
                module_name = f"{rel_path.replace(os.sep, '.')}.{f.stem}"
            else:
                module_name = f.stem
            print(f"Trying to load module: {module_name}")
            module = import_module(module_name)
            try:
                cls = next(filter(lambda c: isinstance(c, type) and issubclass(c, SweepConfig) and c != SweepConfig, map(lambda n: getattr(module, n), filter(lambda n: not n.startswith("__") and not n.endswith("__"), dir(module)))))
                self._config = cls(self.config_file_path)
                print(f"Loaded config from {f.stem}{f.suffix}")
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
                cfg = config(self.config_file_path) # type: ignore
                if getattr(cfg._config['SIMULATOR'], 'TYPE', 'spectre').lower() == sim_name:
                    self._config = cfg
                    print(f"Loaded {sim_name} config from default Config class.")
                    break
        
        self._config._write_netlist()

    @property
    def _simulator(self) -> Simulator:
        return self._config._simulator
    
    def run(self):
        return self._simulator.run()
