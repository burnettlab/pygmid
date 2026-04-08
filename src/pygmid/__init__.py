""" pygmid package """
import glob
import importlib
import logging
import logging.handlers
import sys
from itertools import chain
from pathlib import Path

__all__ = []

# Setup logging
logger = logging.getLogger(__name__)
handlers = [
    logging.StreamHandler(), 
    logging.handlers.RotatingFileHandler(filename=f"{Path(__name__)}.log", maxBytes=1_000_000, backupCount=1),
]
levels = [
    logging.WARNING,
    logging.NOTSET,
]
formatters = [
    logging.Formatter("%(asctime)s %(levelname)s (PyGMID): %(message)s", datefmt="%H:%M:%S"),
    logging.Formatter("%(asctime)s %(levelname)s (PyGMID): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
]

for handler, level, formatter in zip(handlers, levels, formatters, strict=True):
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Import submodules and construct __all__
package_path = Path(__file__).parent

ordered_imports = []
for module in list(map(lambda m: Path(__file__).parent / m, ordered_imports)) + glob.glob(f"{package_path}/*.py") + list(map(lambda s: s.replace("/__init__.py", ""), glob.glob(f"{package_path}/*/__init__.py"))):
    mod_name = str(Path(module).relative_to(package_path).with_suffix('')).replace("/", ".")
    if not mod_name.startswith("__") and not mod_name.endswith("__") and f"{__package__}.{mod_name}" not in sys.modules:
        __all__.append(mod_name)
        mod = importlib.import_module(f".{mod_name}", package=__package__)
        vars().update(filter(lambda e: e[0] in getattr(mod, "__all__", []), vars(mod).items()))

__all__.extend(chain.from_iterable(map(lambda m: getattr(m, "__all__", []), filter(lambda m: getattr(m, "__package__", None) == __package__, vars().copy().values()))))
__version__ = "1.2.12"
