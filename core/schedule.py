"""Execution scheduling primitives -- type-agnostic, pure orchestration.

  * parallel_compile : run many compile tasks across CPU cores (no NPU).
  * distribute_over_cards : run work items across NPU cards with a device queue,
    so each card runs exactly one process at a time while cards run in parallel.

Worker callables must be top-level / picklable (functools.partial of a top-level
function is fine).
"""

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager


def parallel_compile(items, compile_one, workers=None):
    """Run ``compile_one(item)`` for every item on a CPU process pool."""
    if not items:
        return []
    workers = workers or min(len(items), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(compile_one, items))


def _card_task(packed):
    worker, item, queue = packed
    device = queue.get()
    try:
        return worker(item, device)
    finally:
        queue.put(device)


def distribute_over_cards(items, devices, worker):
    """Run ``worker(item, device)`` across ``devices``; one process per card.

    Results are returned in the same order as ``items``.
    """
    if not items:
        return []
    mgr = Manager()
    queue = mgr.Queue()
    for d in devices:
        queue.put(d)
    packed = [(worker, it, queue) for it in items]
    with ProcessPoolExecutor(max_workers=len(devices)) as pool:
        return list(pool.map(_card_task, packed))
