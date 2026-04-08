""" Lookup: pygmid Lookup classes"""
import glob
import importlib
import sys
from itertools import chain
from pathlib import Path

package_path = Path(__file__).parent
__all__ = []

ordered_imports = []
for module in list(map(lambda m: Path(__file__).parent / m, ordered_imports)) + glob.glob(f"{package_path}/*.py") + list(map(lambda s: s.replace("/__init__.py", ""), glob.glob(f"{package_path}/*/__init__.py"))):
    mod_name = str(Path(module).relative_to(package_path).with_suffix('')).replace("/", ".")
    if not mod_name.startswith("__") and not mod_name.endswith("__") and f"{__package__}.{mod_name}" not in sys.modules:
        __all__.append(mod_name)
        mod = importlib.import_module(f".{mod_name}", package=__package__)
        vars().update(filter(lambda e: e[0] in getattr(mod, "__all__", []), vars(mod).items()))

__all__.extend(chain.from_iterable(map(lambda m: getattr(m, "__all__", []), filter(lambda m: getattr(m, "__package__", None) == __package__, vars().copy().values()))))		# type: ignore
