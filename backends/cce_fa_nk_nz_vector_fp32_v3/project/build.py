"""Build a local, self-registering standard CANN SHARED package; never install OPP."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity():
    cann = os.environ.get("ASCEND_HOME_PATH")
    compiler = shutil.which("bisheng")
    if not cann or not compiler:
        raise RuntimeError("source env.sh before building an Ascend C project")
    files = [ROOT / "CMakeLists.txt", Path(__file__)]
    for folder in ("op_host", "op_kernel"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    return {
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
        "cann": str(Path(cann).resolve()),
        "compiler": str(Path(compiler).resolve()),
        "compiler_sha256": digest(Path(compiler)),
        "soc": "ascend910_93",
    }


def directory():
    key = hashlib.sha256(json.dumps(identity(), sort_keys=True).encode()).hexdigest()[:20]
    return ROOT / ".build" / key


def build():
    before = identity()
    out = directory()
    manifest = out / "build.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["identity"] != before:
            raise RuntimeError("standard-project build identity mismatch")
        for name, expected in saved["artifacts"].items():
            if digest(out / name) != expected:
                raise RuntimeError(f"standard-project artifact hash mismatch: {name}")
        return out
    # A new source identity gets a fresh directory. Failed partial builds are not reused.
    if out.exists():
        raise RuntimeError(f"incomplete build remains at {out}; archive it before retrying")
    out.mkdir(parents=True)
    commands = [
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(out),
            "-DENABLE_BINARY_PACKAGE=ON",
            "-DENABLE_SOURCE_PACKAGE=OFF",
            "-DASCEND_SKIP_FAILED_COMPUTE_UNIT=OFF",
        ],
        ["cmake", "--build", str(out), "-j", "8"],
        [
            "cmake",
            "--build",
            str(out),
            "--target",
            "ascendc_kernels_ascendc_bin_ascend910_93_gen_ops_config",
            "-j",
            "8",
        ],
    ]
    for i, command in enumerate(commands):
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (out / f"step{i}.log").write_text(result.stdout)
        print(result.stdout, flush=True)
        if result.returncode or "[ERROR]" in result.stdout or "error:" in result.stdout:
            raise RuntimeError(f"standard-project build failed; see {out / f'step{i}.log'}")
    binary = out / "op_kernel/ascendc_kernels/binary"
    required = [
        out / "libfa_fa_nk_nz_vector_fp32_v3.so",
        out / "op_host/libfa_fa_nk_nz_vector_fp32_v3_ascendc_cust_optiling.so",
        binary / "config/ascend910_93/binary_info_config.json",
    ]
    objects = list((binary / "ascend910_93").rglob("*.o"))
    if not objects or any(not p.is_file() or not p.stat().st_size for p in required + objects):
        raise RuntimeError("standard package is missing fresh kernel/config/library artifacts")
    copied = binary / "dynamic/fa_fa_nk_nz_vector_fp32_v3.cpp"
    if copied.read_bytes() != (ROOT / "op_kernel/fa_fa_nk_nz_vector_fp32_v3.cpp").read_bytes():
        raise RuntimeError("CANN copied a stale kernel source")
    if identity() != before:
        raise RuntimeError("project sources changed during build")
    shutil.copytree(ROOT / "op_host", out / "source/op_host")
    shutil.copytree(ROOT / "op_kernel", out / "source/op_kernel")
    shutil.copy2(ROOT / "CMakeLists.txt", out / "source/CMakeLists.txt")
    shutil.copy2(Path(__file__), out / "source/build.py")
    artifacts = (
        required + objects + list(binary.rglob("*.json")) + list((out / "source").rglob("*"))
    )
    manifest.write_text(
        json.dumps(
            {
                "identity": before,
                "artifacts": {str(p.relative_to(out)): digest(p) for p in artifacts if p.is_file()},
            },
            indent=2,
        )
        + "\n"
    )
    return out


if __name__ == "__main__":
    print(build())
