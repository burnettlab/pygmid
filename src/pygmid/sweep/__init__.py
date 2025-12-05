""" pygmid sweep package """
import glob
import importlib
from itertools import chain
from pathlib import Path
from .__main__ import run


package_path = Path(__file__).parent

for module in glob.glob(f"{package_path}/*.py") + glob.glob(f"{package_path}/*/__init__.py"):
    mod_name = str(Path(module).relative_to(package_path).with_suffix('')).replace("/", ".")
    if not mod_name.startswith("__") and not mod_name.endswith("__"):
        mod = importlib.import_module(f".{mod_name}", package=__package__)
        vars().update(filter(lambda e: e[0] in getattr(mod, "__all__", []), vars(mod).items()))

__all__ = ["run"] + list(chain.from_iterable(map(lambda m: getattr(m, "__all__", []), filter(lambda m: getattr(m, "__package__", None) == __package__, vars().copy().values()))))