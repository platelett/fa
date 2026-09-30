"""In-process device-time profiling via torch_npu.profiler.

profile_case_us() -- one profiler session per case captures all backends
(each has a unique kernel name). Returns per-backend avg Duration(us).
"""

import csv
import glob
import os
import shutil
import tempfile


def _match_backend(name, backend_names, op_types, kernel_names):
    """Match explicit symbols exactly; only external operators use fallback filters."""
    for n in backend_names:
        if name == kernel_names.get(n, f"{n}_kernel"):
            return n
    for n in backend_names:
        if n in kernel_names:
            continue
        op = op_types.get(n, "")
        if (n in name) or (op and op in name):
            return n
    return None


def profile_case_us(calls, backend_names, iters=20, warmup=3, op_types=None,
                    kernel_names=None):
    """Profile all backends in one torch_npu.profiler session.

    ``calls`` is a dict {backend_name: zero-arg callable}.
    ``op_types`` is an optional dict for fallback name matching.
    ``kernel_names`` maps TileLang backends to exact emitted symbols; these
    backends never use substring or op_type fallback matching.
    Returns {backend_name: avg_us or None}.
    """
    import torch
    import torch_npu

    for n in backend_names:
        if n not in calls:
            continue
        for _ in range(warmup):
            calls[n]()
    torch.npu.synchronize()

    tmpdir = tempfile.mkdtemp()
    # Level0 skips the analysis passes we never read (trace/memory/communication
    # views); we only parse kernel_details.csv. Shaves ~1s per session.
    exp_cfg = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0)
    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(tmpdir),
            experimental_config=exp_cfg,
        ):
            for n in backend_names:
                if n not in calls:
                    continue
                for _ in range(iters):
                    calls[n]()
                torch.npu.synchronize()

        kd = glob.glob(os.path.join(tmpdir, "**", "kernel_details.csv"), recursive=True)
        if not kd:
            return {n: None for n in backend_names}

        op_types = op_types or {}
        kernel_names = kernel_names or {}
        by_backend = {}
        with open(kd[0], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("Name", "")
                d = float(row["Duration(us)"])
                n = _match_backend(name, backend_names, op_types, kernel_names)
                if n is not None and n in calls:
                    by_backend.setdefault(n, []).append(d)

        return {n: (sum(ds) / len(ds) if ds else None)
                for n in backend_names
                for ds in [by_backend.get(n, [])]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
