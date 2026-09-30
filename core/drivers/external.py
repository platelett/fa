"""Generic Python-callable adapter, formerly named ascendc.

Optional build(case) must be CPU-safe and idempotent. adapt(case, canonical)
prepares an opaque argument object; call(op_args) returns canonical output.
The wrapped implementation is not restricted to any kernel language or API.
"""

requires_compile = False


def needs_compile(spec):
    return callable(getattr(spec, "build", None))


def compile(spec, case):
    hook = getattr(spec, "build", None)
    if hook is not None:
        if not callable(hook):
            raise TypeError("external build must be callable")
        hook(case)


def make_call(spec, case, canonical):
    compile(spec, case)
    op_args = spec.adapt(case, canonical)
    return lambda: spec.call(op_args)
