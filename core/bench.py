"""Benchmark orchestrator.

Phase A (CPU, parallel)   prepare each backend/case selected by its driver's
                          compile policy; no NPU is touched.
Phase B (NPU, per-card)   fully in-process: one torch_npu.profiler session per
                          case profiles all backends, then correctness checks run
                          in the same process (operators must initialize their own
                          workspace -- the harness does not zero GM). Cards run in
                          parallel.
Report                    long-format CSV + console table: (case x backend) with
                          avg kernel time, ratio-vs-reference, and check verdict.

All backends are handled uniformly via Meta + the type driver; the orchestrator
itself is backend-agnostic. Run under the project env (see env.sh).
"""

import argparse
import csv
import functools
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core import profiler, registry, schedule
from core.contract import KernelNameError
from problem.cases import CASES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO_ROOT, "core", "runner.py")


def _runner(*extra):
    return [sys.executable, RUNNER, *map(str, extra)]


# --------------------------------------------------------------------------- #
# Phase A: parallel compile
# --------------------------------------------------------------------------- #
def _compile_task(packed):
    """Run one backend compile in a fresh CPU process (no NPU)."""
    backend, case_index = packed
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    cmd = _runner("--backend", backend, "--case-index", case_index, "--mode", "compile")
    r = subprocess.run(cmd, env=env, cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return backend, case_index, r.returncode, r.stdout


def compile_phase(backend_names, case_indices):
    tasks = []
    for name in backend_names:
        spec = registry.get(name)
        if not registry.needs_compile(spec):
            continue
        tasks.extend((name, i) for i in case_indices)
    if not tasks:
        return
    print(f">>> Phase A: compiling {len(tasks)} kernels on CPU...")
    failed = []
    for backend, ci, rc, out in schedule.parallel_compile(tasks, _compile_task):
        tag = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"    {backend} {CASES[ci].name}: {tag}")
        if rc != 0:
            print(out)
            failed.append(f"{backend}/{CASES[ci].name}")
    if failed:
        raise RuntimeError(f"Compilation failed for {', '.join(failed)}")


# --------------------------------------------------------------------------- #
# Phase B: fully in-process (one profiler session + checks per case)
# --------------------------------------------------------------------------- #
def process_case(case_index, device, *, session_dir, backend_names, ref_name,
                 iters, do_check, seed, atol, rtol):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    from problem.canonical import build_canonical

    case = CASES[case_index]
    prepared = {}
    skipped = set()
    for n in backend_names:
        try:
            prepared[n] = registry.prepare_call(registry.get(n), case)
        except KernelNameError:
            raise
        except Exception as e:
            print(f"    [skip] {n}: {e}")
            skipped.add(n)
    canonical = build_canonical(case, seed)

    calls = {}
    for n in backend_names:
        if n in skipped:
            continue
        try:
            call = prepared[n](canonical)
            call()
            torch.npu.synchronize()
            calls[n] = call
        except KernelNameError:
            raise
        except Exception as e:
            print(f"    [skip] {n}: {e}")
            skipped.add(n)

    op_types = {n: registry.get(n).META.op_type for n in backend_names}
    kernel_names = {n: getattr(calls.get(n), "kernel_name", f"{n}_kernel")
                    for n in backend_names
                    if registry.get(n).META.type == "tilelang"}
    times = profiler.profile_case_us(calls, backend_names, iters=iters,
                                     op_types=op_types, kernel_names=kernel_names)

    # Correctness checks run in-process, reusing the already-built callables.
    # This assumes each operator initializes its own workspace and does not
    # depend on GM being zeroed -- a kernel that reads uninitialized workspace
    # is a kernel bug to fix there, not something the harness works around.
    checks = {}
    if do_check:
        ref_out = None
        ordered = sorted(backend_names, key=lambda n: not registry.get(n).META.is_reference)
        for n in ordered:
            if n in skipped:
                checks[n] = ("ref", 0.0) if registry.get(n).META.is_reference else ("ERR", None)
                continue
            out = calls[n]().float()
            torch.npu.synchronize()
            if registry.get(n).META.is_reference:
                ref_out = out
                checks[n] = ("ref", 0.0)
            elif ref_out is not None:
                max_diff = (out - ref_out).abs().max().item()
                try:
                    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
                    checks[n] = ("PASS", max_diff)
                except AssertionError:
                    checks[n] = ("FAIL", max_diff)
            else:
                checks[n] = ("ERR", None)

    ref_us = times.get(ref_name)
    rows = []
    for n in backend_names:
        meta = registry.get(n).META
        us = times.get(n)
        ratio = f"{(us / ref_us) * 100:.1f}%" if (us and ref_us) else "N/A"
        status, diff = checks.get(n, ("skip", None))
        rows.append([case.name, n, "ref" if meta.is_reference else "",
                     f"{us:.2f}" if us else "N/A", ratio, status,
                     f"{diff:.5f}" if diff is not None else "N/A"])
    print(f"[done] {case.name:<16} dev={device} " +
          " ".join(f"{n}={times[n]:.1f}us" if times.get(n) else f"{n}=N/A" for n in backend_names))
    return rows


# --------------------------------------------------------------------------- #
def _default_devices():
    env = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if env.strip():
        return [int(x) for x in env.split(",") if x.strip()]
    return list(range(16))


def _parse_cases(spec):
    spec = spec.strip()
    if spec == "all":
        return list(range(len(CASES)))
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            i = int(tok)
        except ValueError:
            raise SystemExit(f"--cases: '{tok}' is not an integer index or 'all'")
        if not (0 <= i < len(CASES)):
            raise SystemExit(f"--cases: index {i} out of range 0..{len(CASES) - 1}")
        if i not in out:
            out.append(i)
    if not out:
        raise SystemExit("--cases: no valid case indices given")
    return out


def main():
    p = argparse.ArgumentParser(description="benchmark orchestrator")
    p.add_argument("--backends", type=str, default=None,
                   help="comma-separated backend names (default: all registered)")
    p.add_argument("--cases", type=str, default="all",
                   help="'all' (default) or comma-separated case indices, e.g. 0,3,6 "
                        "(use --list-cases for the index<->name table)")
    p.add_argument("--list-cases", action="store_true",
                   help="print the case index<->name table and exit")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--logdir", type=str, default=os.path.join(REPO_ROOT, "results"))
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--rtol", type=float, default=1e-2)
    args = p.parse_args()

    if args.list_cases:
        for i, c in enumerate(CASES):
            print(f"{i:2d}  {c.name}")
        return

    backend_names = (args.backends.split(",") if args.backends else registry.all_names())
    for name in backend_names:
        registry.require_attention(registry.get(name))
    ref_name, _ = registry.get_reference()
    if ref_name not in backend_names:
        backend_names = [ref_name] + backend_names

    case_indices = _parse_cases(args.cases)
    devices = ([int(x) for x in args.devices.split(",")] if args.devices else _default_devices())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "all" if args.cases.strip() == "all" else "sel" + "_".join(map(str, case_indices))
    session_dir = os.path.join(args.logdir, f"bench_{label}_{stamp}")
    os.makedirs(session_dir, exist_ok=True)
    print(f">>> backends={backend_names} ref={ref_name}")
    print(f">>> cases: {len(case_indices)}/{len(CASES)} -> indices {case_indices} on devices {devices}")
    print(f">>> session: {session_dir}")

    if not args.no_compile:
        compile_phase(backend_names, case_indices)

    print(f">>> Phase B: {len(case_indices)} cases on {len(devices)} cards...")
    worker = functools.partial(process_case, session_dir=session_dir, backend_names=backend_names,
                               ref_name=ref_name, iters=args.iters,
                               do_check=not args.no_check,
                               seed=args.seed, atol=args.atol, rtol=args.rtol)
    per_case_rows = schedule.distribute_over_cards(case_indices, devices, worker)

    headers = ["Case", "Backend", "Ref", "Avg_us", "vs_ref", "Check", "MaxDiff"]
    rows = [r for case_rows in per_case_rows for r in case_rows]
    csv_path = os.path.join(session_dir, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    print("\n" + "  ".join(headers))
    for r in rows:
        print("  ".join(str(x) for x in r))
    print(f"\n[done] report -> {csv_path}")


if __name__ == "__main__":
    main()
