"""pygmid"""

import importlib
import sys
from itertools import chain
from pathlib import Path

__all__ = []


# Import wanted submodules and construct __all__
package_path = Path(__file__).parent

wanted_imports = ["logging", "lookup"]
for module in map(lambda m: Path(__file__).parent / m, wanted_imports):
    mod_name = str(Path(module).relative_to(package_path).with_suffix("")).replace(
        "/", "."
    )
    if f"{__package__}.{mod_name}" in sys.modules:
        mod = sys.modules[f"{__package__}.{mod_name}"]
    else:
        mod = importlib.import_module(f".{mod_name}", package=__package__)

    vars().update(
        filter(lambda e: e[0] in getattr(mod, "__all__", []), vars(mod).items())
    )

__all__.extend(
    chain.from_iterable(
        map(
            lambda m: getattr(m, "__all__", []),
            filter(
                lambda m: getattr(m, "__package__", None) == __package__,
                vars().copy().values(),
            ),
        )
    )
)
__all__ = list(set(__all__))
