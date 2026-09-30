"""Backend metadata + the contracts the framework talks to.

Two roles, kept apart:

  * A *backend* is a module under backends/<name>/backend.py exposing ``META``
    and type-specific declarations/adapter hooks. External adapters may bridge
    an existing build or calling API; standard projects declare their recipe.

  * A *driver* (core/drivers/<type>.py) owns all the type-specific dirty work --
    turning a backend declaration into something runnable, and compiling it.
    The framework picks the driver by ``META.type``.

The framework (runner/bench) only ever uses ``META`` + the driver, so it is fully
backend-agnostic.

Backend module contract by type:
  type == "tilelang":  kernel(case, *, kernel_name: str) -> JITKernel
                         build the operator from the problem sizes only (shape-only,
                         cacheable, safe without an NPU). The kernel auto-allocates
                         its Output + scratch/workspace (out_idx / workspace_idx).
                         Set global_symbol to kernel_name before JIT compilation;
                         the driver supplies the registry's backend directory name
                         and checks the result on both compile and run paths.
                       assemble(case, canonical) -> args_tuple
                         map the shared canonical tensors to the kernel's positional
                         args (needs the real NPU tensors); returned output is already
                         in canonical shape.
  type == "external": optional idempotent build(case); adapt(case, canonical)
                       -> opaque op_args; call(op_args) -> canonical output
  type == "ascendc" :  project(case) -> core.drivers.ascendc.Project
                       load(case, project) -> built operator callable
                       adapt(case, canonical) -> positional args tuple
                       Requires a standard op_host/op_kernel source project,
                       including when the device implementation uses raw CCE.

Driver module contract:
  requires_compile: bool
  needs_compile(spec) -> bool   # optional per-backend override
  compile(spec, case) -> None
  prepare(spec, case) -> bind(canonical)   # optional early package registration
  make_call(spec, case, canonical) -> Callable[[], Tensor]   # one kernel/op run
"""

from dataclasses import dataclass


class KernelNameError(ValueError):
    """A backend violated the harness's compiled-symbol contract."""


@dataclass(frozen=True)
class Meta:
    name: str                    # legacy; framework keys backends by directory name
    type: str                    # "tilelang" | "external" | "ascendc"
    op_type: str                 # fallback kernel-name filter for profiling (used when
                                 # the profiled kernel name isn't the directory name,
                                 # e.g. external ops like FlashAttentionScore)
    is_reference: bool = False   # exactly one backend is the correctness reference
    author: str = ""
    date: str = ""
    notes: str = ""
    diagnostic: bool = False     # not an attention implementation; excluded from FA checks
