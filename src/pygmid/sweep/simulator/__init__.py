from .sim import Simulator
from .spectre import SpectreSimulator
from .ngspice import NGSpiceSimulator


SIMULATORS = {
    'spectre': SpectreSimulator,
    'ngspice': NGSpiceSimulator,
}

__all__ = ['SIMULATORS', "Simulator"]
