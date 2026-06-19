"""Benchmark: parallelism speed-up of splitting work across parallel Flow branches.

A task needs several *independent* CPU-bound sub-computations. We compare two
flow topologies that do the exact same total work:

  * sequential: a single handler computes all sub-results one after another;
  * parallel:   N branch handlers each compute one sub-result concurrently
                (the task is fanned out, then the results are collected/merged).

Reported metrics:
  * per-task latency speed-up (concurrency = 1): isolates how much faster a
    single task finishes when its work runs across N processes instead of one;
  * throughput speed-up (concurrency > 1): tasks/sec under load.

Work is a pure-Python integer busy loop (NOT numpy/BLAS): it is genuinely
single-threaded within a process, so splitting it across N branch processes
exposes real process-level parallelism. (A numpy matmul would be a poor choice
here because BLAS — e.g. Apple Accelerate / OpenBLAS — already multithreads a
single call and saturates the cores, leaving nothing for the branches to gain.)

Run:  python -m tests.benchmarks.bench_parallel_speedup
"""
import asyncio
import multiprocessing as mp
import statistics
import time
from typing import Dict, List, Type

from aqueduct.flow import Flow
from aqueduct.handler import BaseTaskHandler
from aqueduct.task import BaseTask


# Tuning knobs. Work per task is deliberately heavy so that compute (not the
# per-task IPC/process-hop overhead) dominates; otherwise fan-out overhead masks
# the parallel speed-up. Lower ITER_PER_UNIT for a quicker (but noisier) run.
ITER_PER_UNIT = 1_500_000  # pure-Python iterations per work unit (~tens of ms)
TOTAL_UNITS = 8            # total independent sub-computations per task
N_TASKS = 12               # tasks processed per measurement
TIMEOUT_SEC = 120.0        # generous per-task timeout for slow machines


def _cpu_work(units: int) -> int:
    """Deterministic single-threaded CPU busywork; returns a checksum.

    Uses a pure-Python LCG loop (no numpy) so the work is single-threaded and
    unaffected by BLAS multithreading. The checksum is returned (and stored on
    the task) so the computation cannot be optimised away.
    """
    acc = 0
    for _ in range(units):
        x = 1
        for _ in range(ITER_PER_UNIT):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        acc += x
    return acc


class Payload(BaseTask):
    def __init__(self):
        super().__init__()
        self.result = None


def _make_work_handler(name: str, units: int, attr: str) -> Type[BaseTaskHandler]:
    """Build a handler class that does ``units`` of CPU work and stores the result."""
    class _WorkHandler(BaseTaskHandler):
        def handle(self, *tasks: BaseTask):
            for task in tasks:
                setattr(task, attr, _cpu_work(units))

    _WorkHandler.__name__ = name
    _WorkHandler.__qualname__ = name
    return _WorkHandler


def _build_sequential_flow(total_units: int) -> Flow:
    handler_cls = _make_work_handler('SeqWork', total_units, 'result')
    return Flow(handler_cls())


def _build_parallel_flow(total_units: int, branches: int) -> Flow:
    per_branch = max(1, total_units // branches)
    steps = [
        _make_work_handler(f'ParWork{i}', per_branch, f'r{i}')()
        for i in range(branches)
    ]
    return Flow(steps)  # list of handlers -> parallel group + auto collector


async def _measure(flow: Flow, n_tasks: int, concurrency: int) -> Dict[str, float]:
    flow.start()
    try:
        # Warm up forks / BLAS / allocator before timing.
        await flow.process(Payload(), timeout_sec=TIMEOUT_SEC)

        latencies: List[float] = []
        semaphore = asyncio.Semaphore(concurrency)

        async def _one():
            async with semaphore:
                start = time.perf_counter()
                await flow.process(Payload(), timeout_sec=TIMEOUT_SEC)
                latencies.append(time.perf_counter() - start)

        wall_start = time.perf_counter()
        await asyncio.gather(*[_one() for _ in range(n_tasks)])
        wall = time.perf_counter() - wall_start
    finally:
        await flow.stop(graceful=False)

    return {
        'wall': wall,
        'throughput': n_tasks / wall,
        'mean_latency': statistics.mean(latencies),
        'p50_latency': statistics.median(latencies),
    }


async def _run_suite(branches_options: List[int]) -> None:
    print(f'config: ITER_PER_UNIT={ITER_PER_UNIT}, TOTAL_UNITS={TOTAL_UNITS}, '
          f'N_TASKS={N_TASKS}, cpu_count={mp.cpu_count()}\n')

    # Baseline: single sequential handler.
    seq_latency = await _measure(_build_sequential_flow(TOTAL_UNITS), N_TASKS, concurrency=1)
    seq_throughput = await _measure(_build_sequential_flow(TOTAL_UNITS), N_TASKS,
                                    concurrency=max(branches_options))

    print('--- per-task latency (concurrency=1) ---')
    print(f'{"topology":<16}{"mean ms":<12}{"p50 ms":<12}{"latency speed-up":<18}')
    print(f'{"sequential":<16}{seq_latency["mean_latency"]*1e3:<12.1f}'
          f'{seq_latency["p50_latency"]*1e3:<12.1f}{1.0:<18.2f}')

    latency_rows = []
    throughput_rows = []
    for branches in branches_options:
        par_latency = await _measure(_build_parallel_flow(TOTAL_UNITS, branches),
                                     N_TASKS, concurrency=1)
        speedup = seq_latency['mean_latency'] / par_latency['mean_latency']
        latency_rows.append((branches, par_latency, speedup))
        print(f'{f"parallel x{branches}":<16}{par_latency["mean_latency"]*1e3:<12.1f}'
              f'{par_latency["p50_latency"]*1e3:<12.1f}{speedup:<18.2f}')

    print('\n--- throughput (concurrency = #branches) ---')
    print(f'{"topology":<16}{"tasks/sec":<14}{"throughput speed-up":<20}')
    print(f'{"sequential":<16}{seq_throughput["throughput"]:<14.2f}{1.0:<20.2f}')
    for branches in branches_options:
        par_throughput = await _measure(_build_parallel_flow(TOTAL_UNITS, branches),
                                        N_TASKS, concurrency=branches)
        speedup = par_throughput['throughput'] / seq_throughput['throughput']
        throughput_rows.append((branches, par_throughput, speedup))
        print(f'{f"parallel x{branches}":<16}{par_throughput["throughput"]:<14.2f}'
              f'{speedup:<20.2f}')


def main():
    branches_options = [2, 4]
    if mp.cpu_count() >= 8:
        branches_options.append(8)
    asyncio.run(_run_suite(branches_options))


if __name__ == '__main__':
    mp.set_start_method('fork')
    main()
