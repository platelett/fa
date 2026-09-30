"""Backend discovery + driver lookup.

A backend is a module ``backends/<dir>/backend.py`` exposing a module-level
``META`` (core.contract.Meta) and the type-specific core-logic functions. The
registry keys backends by their directory name and maps ``META.type`` to the
matching driver module. Importing a backend module must stay cheap (heavy/NPU
imports belong inside the functions, not at module top), so discovery is safe to
run anywhere -- including on a host with no NPU.
"""

import importlib
import os

_BACKENDS = None  # dirname -> module
_DRIVERS = {
    "tilelang": "core.drivers.tilelang",
    "external": "core.drivers.external",
    "ascendc":  "core.drivers.ascendc",
}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_call(spec, case):
    """Prepare early-loading drivers before creating any canonical NPU tensors."""
    driver = driver_for(spec.META)
    prepare = getattr(driver, "prepare", None)
    if prepare is not None:
        return prepare(spec, case)
    return lambda canonical: driver.make_call(spec, case, canonical)


def discover(force=False):
    global _BACKENDS
    if _BACKENDS is not None and not force:
        return _BACKENDS
    backends = {}
    backends_dir = os.path.join(_repo_root(), "backends")
    for dirname in sorted(os.listdir(backends_dir)):
        if not os.path.isfile(os.path.join(backends_dir, dirname, "backend.py")):
            continue
        backends[dirname] = importlib.import_module(f"backends.{dirname}.backend")
    _BACKENDS = backends
    return backends


def get(name):
    """Return the backend module for ``name``."""
    return discover()[name]


def all_names(include_diagnostics=False):
    return [name for name, spec in discover().items()
            if include_diagnostics or not spec.META.diagnostic]


def require_attention(spec):
    if spec.META.diagnostic:
        name = name_of(spec)
        raise ValueError(f"{name} is diagnostic-only, not FA; use python -m backends.{name}.bench")


def name_of(spec):
    """Return the registry key of a backend module, independent of legacy META.name."""
    names = [name for name, module in discover().items() if module is spec]
    if len(names) != 1:
        raise ValueError(f"expected one registered name for {spec!r}, found {names!r}")
    return names[0]


def get_reference():
    """Return (name, module) of the single is_reference backend."""
    refs = [(n, m) for n, m in discover().items() if m.META.is_reference]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one reference backend, found {len(refs)}")
    return refs[0]


def driver_for(meta):
    return importlib.import_module(_DRIVERS[meta.type])


def needs_compile(spec):
    driver = driver_for(spec.META)
    predicate = getattr(driver, "needs_compile", None)
    return bool(predicate(spec) if predicate else getattr(driver, "requires_compile", False))
