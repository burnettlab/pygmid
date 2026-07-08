"""Pygmid Adapter for PySpice Mosfet"""

import logging
from functools import wraps
from typing import Any, Optional

import PySpice.Spice.BasicElement as BasicElement
from PySpice.Spice.Netlist import Netlist

from .lookup import Lookup

LOGGER = logging.getLogger(__name__)


def convert_lookup(cls):
    cls_init = cls.__init__

    @wraps(cls_init)
    def wrapper(obj, netlist: Netlist, *args, model: Optional[Any] = None, **kwargs):
        if model and isinstance(model, Lookup):
            for inc in model["INCLUDE"]:
                LOGGER.debug(f"include: {inc}")
                if inc not in netlist._includes:
                    netlist.include(inc)

            for lib, section in model["LIB"]:
                LOGGER.debug(f"lib: {lib}, section: {section}")
                if (lib, section) not in netlist._libs:
                    netlist.lib(lib, section=section)

            # Lastly, convert to the model string
            model = model["MODEL"]

        return cls_init(obj, netlist, *args, model=model, **kwargs)

    LOGGER.debug(f"wrapping {cls_init.__name__}")
    cls.__init__ = wrapper
    return wrapper


convert_lookup(BasicElement.Mosfet)
