from .sim import Simulator
from .spectre import SpectreSimulator
from .ngspice import ngspiceSimulator


SIMULATORS = {
    'spectre': SpectreSimulator,
    'ngspice': ngspiceSimulator,
}

__all__ = ['SIMULATORS', "Simulator"]
