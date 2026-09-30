"""Single (backend x case) execution body, run as an isolated subprocess.

Modes:
  compile : prepare/build the backend through its driver; no NPU execution.
  check   : run the backend + the reference once and compare outputs. Each check
            runs in its own process because distinct tilelang kernels sharing a
            process can corrupt each other's results (a kernel that reads its
            workspace before initializing it sees another kernel's leftover
            memory); a fresh process gives clean, driver-zeroed device memory.

Heavy/NPU imports stay inside main() so the compile path stays light.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_args():
    p = argparse.ArgumentParser(description="single runner")
    p.add_argument("--backend", required=True)
    p.add_argument("--case-index", type=int, required=True)
    p.add_argument("--mode", choices=("compile", "check"), default="check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--rtol", type=float, default=1e-2)
    return p.parse_args()


def main():
    args = _parse_args()

    from core import registry
    from problem.cases import CASES

    case = CASES[args.case_index]
    spec = registry.get(args.backend)
    registry.require_attention(spec)
    meta = spec.META
    driver = registry.driver_for(meta)

    if args.mode == "compile":
        driver.compile(spec, case)
        print(f"COMPILE {args.backend} {case.name} done")
        return

    import torch
    from problem.canonical import build_canonical

    bind = registry.prepare_call(spec, case)
    if not meta.is_reference:
        ref_name, ref_spec = registry.get_reference()
        ref_bind = registry.prepare_call(ref_spec, case)
    canonical = build_canonical(case, args.seed)
    out = bind(canonical)().float()

    if meta.is_reference:
        print(f"CHECK {case.name} backend={args.backend} PASS max_abs_diff=0.00000 (self)")
        return

    ref_out = ref_bind(canonical)().float()
    torch.npu.synchronize()
    max_diff = (out - ref_out).abs().max().item()
    try:
        torch.testing.assert_close(out, ref_out, atol=args.atol, rtol=args.rtol)
        status = "PASS"
    except AssertionError:
        status = "FAIL"
    print(f"CHECK {case.name} backend={args.backend} {status} "
          f"max_abs_diff={max_diff:.5f} (atol={args.atol}, rtol={args.rtol})")


if __name__ == "__main__":
    main()
