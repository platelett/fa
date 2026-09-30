"""tilelang driver -- all tilelang-type dirty work (compile + launch shape).

Thin: caching is handled by @tilelang.jit inside the backend's kernel module, and
profiling/scheduling live in the framework. The driver only encodes *how* a tilelang
backend is compiled and turned into a single callable, using the two functions a
tilelang backend exposes:

  * ``kernel(case, *, kernel_name) -> JITKernel``
                                      build with the registered backend name
                                      (shape-only; cacheable; safe on a host w/o NPU)
  * ``assemble(case, canonical)``     map the shared canonical tensors to this
        -> args_tuple                 kernel's positional args (needs the NPU data)

The kernel auto-allocates its own Output + scratch/workspace (out_idx/workspace_idx)
and returns the output already in canonical shape, so the driver just calls it.
"""

import inspect
import re

from core import registry
from core.contract import KernelNameError

requires_compile = True


def _build(spec, case):
    kernel_name = registry.name_of(spec)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kernel_name) is None:
        raise KernelNameError(f"Backend name {kernel_name!r} is not a valid kernel symbol")

    try:
        inspect.signature(spec.kernel).bind(case, kernel_name=kernel_name)
    except TypeError as exc:
        raise KernelNameError(
            f"Backend {kernel_name!r} must accept kernel(case, *, kernel_name)"
        ) from exc

    func = spec.kernel(case, kernel_name=kernel_name)
    attrs = getattr(getattr(func, "prim_func", None), "attrs", None)
    actual = str(attrs["global_symbol"]) if attrs is not None and "global_symbol" in attrs else None
    if actual != kernel_name:
        raise KernelNameError(
            f"Backend {kernel_name!r} compiled symbol {actual!r}; expected {kernel_name!r}. "
            'Apply T.func_attr({"global_symbol": kernel_name}) before JIT compilation.'
        )
    return func


def compile(spec, case):
    """Build + disk-cache the kernel (CPU only); building the JIT func is enough.

    This is why ``kernel`` must depend on sizes alone: the whole compile phase runs
    here, in parallel, on a host that may have no NPU and no input tensors."""
    _build(spec, case)


def make_call(spec, case, canonical):
    """One ready-to-run kernel call: build (cache hit), assemble its args, invoke."""
    func = _build(spec, case)
    args = spec.assemble(case, canonical)
    return lambda: func(*args)
