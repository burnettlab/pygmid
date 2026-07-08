"""pygmid luts package"""

import glob
import importlib
import sys
from typing import Type
from dataclasses import MISSING, Field, fields
from pathlib import Path

from .base import _BaseLUT

LUTS: dict[str, Type[_BaseLUT]] = {}

__all__ = ["LUTS"]

# Import submodules
package_path = Path(__file__).parent

ordered_imports = ["base"]
for module in (
    list(map(lambda m: Path(__file__).parent / m, ordered_imports))
    + glob.glob(f"{package_path}/*.py")
    + list(
        map(
            lambda s: s.replace("/__init__.py", ""),
            glob.glob(f"{package_path}/*/__init__.py"),
        )
    )
):
    mod_name = str(Path(module).relative_to(package_path).with_suffix("")).replace(
        "/", "."
    )
    if not mod_name.startswith("__") and not mod_name.endswith("__"):
        if f"{__package__}.{mod_name}" in sys.modules:
            mod = sys.modules[f"{__package__}.{mod_name}"]
        else:
            mod = importlib.import_module(f".{mod_name}", package=__package__)

        vars().update(
            filter(lambda e: e[0] in getattr(mod, "__all__", []), vars(mod).items())
        )


for name, obj in filter(lambda it: isinstance(it[1], type), vars().copy().items()):
    if issubclass(obj, _BaseLUT) and obj != _BaseLUT:
        var = next(filter(lambda v: v.name == "extensions", fields(obj)))

        # Get the default value (or from factory) for the extensions field (from the first class in the MRO where it's defined)
        default = MISSING
        for ancestor in obj.__mro__:
            if fields_map := ancestor.__dict__.get("__dataclass_fields", {}):
                if field_obj := fields_map.get(var.name, None):
                    if field_obj.default is not MISSING:
                        default = field_obj.default
                        break

                    if field_obj.default_factory is not MISSING:
                        default = field_obj.default_factory()
                        break

            if value := ancestor.__dict__.get(var.name, None):
                if isinstance(value, Field):
                    if value.default is not MISSING:
                        default = value.default
                        break

                    if value.default_factory is not MISSING:
                        default = value.default_factory()
                        break
                default = value
                break

        if default is MISSING:
            raise TypeError(
                f"Subclass '{obj.__name__}' must define a default for '{var.name}' "
                f"either as a dataclass field or a raw value (e.g., float, tuple)."
            )
        assert (
            default
        ), f"Subclass '{obj.__name__}' must define a set of valid extensions (e.g., str, Tuple[str, ...])"
        if isinstance(default, str):
            default = (default,)

        LUTS.update({ext: obj for ext in default})
        __all__.append(name)
