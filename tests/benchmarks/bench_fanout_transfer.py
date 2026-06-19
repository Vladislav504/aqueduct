"""Benchmark: cost of fanning a single task out to N parallel branches.

This isolates the data-transfer concern that dominates on real hardware: when a
task is sent to N parallel branch queues, the plain ``Queue.put`` path pickles
and copies the whole payload N times, while the zero-copy path moves the payload
into shared memory once and sends only a lightweight handle to each branch.

What it demonstrates (valid on CPU, ready for GPU):
  * copy-per-branch total transfer time grows ~linearly with N and payload size;
  * zero-copy total transfer time stays ~flat in N (one SHM copy + N small puts).

On GPU hardware the same shared-memory handle path avoids N host<->device round
trips; the relative behaviour measured here on CPU carries over.

Run:  python -m tests.benchmarks.bench_fanout_transfer
"""
import multiprocessing as mp
import time
from typing import List

import numpy as np

from aqueduct.fanout import prepare_task_for_fanout
from aqueduct.metrics.queue import TaskMetricsQueue
from aqueduct.task import BaseTask


class ImageTask(BaseTask):
    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image


def _branch_worker(q: TaskMetricsQueue, n_items: int, results: mp.Queue):
    """Reads n_items tasks, measures receive+attach time, reports the total."""
    total = 0.0
    for _ in range(n_items):
        start_wall, task = q.get()
        # touch the data to force attach / materialisation (host read)
        _ = int(task.image[0])
        total += time.monotonic() - start_wall
        task = None  # allow gc / shm release promptly
    results.put(total)


def _bench(payload_mb: float, fanout: int, iterations: int, zero_copy: bool) -> dict:
    payload_bytes = int(payload_mb * 1024 * 1024)
    image = np.zeros(payload_bytes, dtype=np.uint8)

    branch_queues: List[TaskMetricsQueue] = [TaskMetricsQueue(maxsize=iterations + 2) for _ in range(fanout)]
    result_queue: mp.Queue = mp.Queue()

    procs = [
        mp.Process(target=_branch_worker, args=(q, iterations, result_queue))
        for q in branch_queues
    ]
    for p in procs:
        p.start()

    put_total = 0.0
    for _ in range(iterations):
        task = ImageTask(image.copy())
        if zero_copy:
            prepare_task_for_fanout(task, fanout_degree=fanout, min_share_bytes=64 * 1024)

        t0 = time.monotonic()
        for q in branch_queues:
            q.put((time.monotonic(), task))
        put_total += time.monotonic() - t0
        task = None
        time.sleep(0.001)

    recv_total = sum(result_queue.get() for _ in range(fanout))

    for p in procs:
        p.join()

    return {
        'payload_mb': payload_mb,
        'fanout': fanout,
        'zero_copy': zero_copy,
        'put_ms_per_iter': (put_total / iterations) * 1e3,
        'recv_ms_per_item': (recv_total / (iterations * fanout)) * 1e3,
    }


def main():
    iterations = 20
    print(f'{"mode":<12}{"payload_mb":<12}{"fanout":<8}{"put ms/iter":<14}{"recv ms/item":<14}')
    for payload_mb in (1.0, 8.0):
        for fanout in (2, 4, 8):
            for zero_copy in (False, True):
                r = _bench(payload_mb, fanout, iterations, zero_copy)
                mode = 'zero-copy' if zero_copy else 'copy'
                print(
                    f'{mode:<12}{r["payload_mb"]:<12}{r["fanout"]:<8}'
                    f'{r["put_ms_per_iter"]:<14.4f}{r["recv_ms_per_item"]:<14.4f}'
                )


if __name__ == '__main__':
    mp.set_start_method('fork')
    main()
