"""Standard Ascend C operator-project driver, not a generic callback wrapper.

Backend contract and project examples are documented in README.md.
The project owns its CANN operator schema/host contract. This driver validates
project structure and build outputs; it does not infer operator semantics.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from pathlib import Path
import subprocess

requires_compile = True


@dataclass(frozen=True)
class Project:
    """Explicit standard-project recipe; commands are argv, never shell strings.

    Empty build_command means use already-built artifacts in the source project.
    Build commands must be CPU-safe and incremental/idempotent. No automatic
    package installation or system-wide OPP mutation is performed by the driver.
    """

    root: Path
    artifacts: tuple[str, ...]
    build_command: tuple[str, ...] = ()
    timeout_seconds: float = 600

    def validate(self):
        root = Path(self.root).resolve()
        if not (root / "CMakeLists.txt").is_file():
            raise ValueError(f"Ascend C project needs CMakeLists.txt: {root}")
        for directory in ("op_host", "op_kernel"):
            folder = root / directory
            if not folder.is_dir() or not any(
                p.is_file() and p.suffix in (".cpp", ".cc", ".cxx") for p in folder.rglob("*")
            ):
                raise ValueError(f"Ascend C project needs {directory} sources: {root}")
        if not isinstance(self.build_command, tuple) or any(
            not isinstance(arg, str) or not arg for arg in self.build_command
        ):
            raise TypeError("build_command must be a tuple of nonempty argv strings")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ValueError("Ascend C project must declare its build artifacts")
        for item in self.artifacts:
            if not isinstance(item, str) or not item or Path(item).is_absolute():
                raise ValueError("artifacts must be nonempty project-relative paths")
            if not (root / item).resolve().is_relative_to(root):
                raise ValueError(f"artifact escapes project root: {item}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return root

    def artifact_paths(self):
        root = self.validate()
        return tuple(root / item for item in self.artifacts)


def _project(spec, case):
    for method in ("project", "load", "adapt"):
        if not callable(getattr(spec, method, None)):
            raise TypeError(
                f"ascendc backend requires {method}(); generic adapt/call backends "
                'must use type="external"'
            )
    project = spec.project(case)
    if not isinstance(project, Project):
        raise TypeError("project(case) must return core.drivers.ascendc.Project")
    project.validate()
    return project


@contextmanager
def _prepared(spec, case):
    project = _project(spec, case)
    root = Path(project.root).resolve()
    # Different cases may build the same standard project in separate workers.
    with (root / ".fa_ascendc.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if project.build_command:
            with (root / ".fa_ascendc_build.log").open("w") as log:
                subprocess.run(
                    list(project.build_command),
                    cwd=root,
                    stdout=log,
                    stderr=log,
                    check=True,
                    timeout=project.timeout_seconds,
                )
        missing = [str(p) for p in project.artifact_paths() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Ascend C build artifacts missing: {missing}")
        yield project


def compile(spec, case):
    with _prepared(spec, case):
        pass


def prepare(spec, case):
    """Load/register the standard package before canonical inputs trigger NPU ops."""
    with _prepared(spec, case) as project:
        operator = spec.load(case, project)
    if not callable(operator):
        raise TypeError("load(case, project) must return the built operator's callable binding")

    def bind(canonical):
        op_args = spec.adapt(case, canonical)
        if not isinstance(op_args, tuple):
            raise TypeError("ascendc adapt(case, canonical) must return a positional-argument tuple")
        return lambda: operator(*op_args)

    return bind


def make_call(spec, case, canonical):
    return prepare(spec, case)(canonical)
