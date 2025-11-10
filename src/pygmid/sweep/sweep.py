import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from warnings import warn

from .config import Config, SweepConfig
from .simulator import Simulator


@dataclass
class Sweep:
    config_file_path: str
    _config: SweepConfig = field(init=False, repr=False)
    
    def __post_init__(self):
        for f in filter(lambda p: p.suffix == ".py", map(lambda p: Path(p), os.listdir(os.getcwd()))):
            # Import the file and check if it has a class that is a subclass of Config
            module_name = f.stem
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
            self._config = Config(self.config_file_path)
        
        self._config._write_netlist()

    @property
    def _simulator(self) -> Simulator:
        return self._config._simulator
    
    def run(self):
        return self._simulator.run()
