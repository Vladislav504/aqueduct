import asyncio
import multiprocessing as mp
import operator
import os
import psutil
import queue
import signal
import sys
from enum import Enum
from functools import cached_property
from multiprocessing import Barrier
from threading import BrokenBarrierError
from time import monotonic
from typing import Dict, List, Literal, Optional, Tuple, Union

from concurrent.futures import ThreadPoolExecutor
from multiprocessing.resource_tracker import _resource_tracker

from .exceptions import FlowError, MPStartMethodValueError, NotRunningError
from .handler import BaseTaskHandler, HandleConditionType, ParallelTasksCollectorHandler
from .logger import log
from .metrics import MAIN_PROCESS, MetricsTypes
from .metrics.collect import Collector, TasksStats
from .metrics.export import Exporter
from .metrics.manager import get_metrics_manager
from .metrics.processes import ProcessesStats
from .metrics.queue import TaskMetricsQueue
from .metrics.timer import timeit
from .multiprocessing import (
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    start_processes,
)
from .queues import FlowStepQueue, select_next_queue
from .task import BaseTask, DEFAULT_PRIORITY, StopTask
from .worker import Worker
from .metrics.base import MetricsItems

# just for using common ResourceTracker in main and child processes and avoiding
# unnecessary shared memory resource_tracker "No such file or directory" warnings
_resource_tracker.ensure_running()


def _check_env():
    if not sys.version_info >= (3, 8):
        raise RuntimeError('Requires python 3.8 or higher to use multiprocessing.shared_memory')


class FlowStep:

    def __init__(
            self,
            handler: BaseTaskHandler,
            handle_condition: HandleConditionType = operator.truth,
            nprocs: int = 1,
            batch_size: int = 1,
            batch_timeout: float = 0,
            on_start_wait: float = 0,
    ):
        _check_env()
        self.handler = handler
        self.handle_condition = handle_condition
        self.nprocs = nprocs
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.on_start_wait = on_start_wait


class FlowState(Enum):
    RUNNING = 'running'
    STARTING = 'starting'
    STOPPING = 'stopping'
    STOPPED = 'stopped'


class Flow:

    def __init__(
            self,
            *steps: Union[FlowStep, List[FlowStep], BaseTaskHandler, List[BaseTaskHandler]],
            metrics_enabled: bool = True,
            metrics_collector: Collector = None,
            metrics_exporter: Exporter = None,
            queue_size: Optional[int] = None,
            queue_priorities: int = 1,
            mp_start_method: Literal['fork', 'spawn', 'forkserver'] = 'fork',
    ):
        _check_env()
        
        # Initialize common Flow attributes
        self._contexts: Dict[BaseTaskHandler, ProcessContext] = {}
        self._queue_priorities = queue_priorities
        self._task_futures: Dict[str, asyncio.Future] = {}
        self._queue_size: Optional[int] = queue_size
        self._tasks: List[asyncio.Future] = []
        self._state: FlowState = FlowState.STOPPED

        if mp_start_method != "fork" and mp_start_method != mp.get_start_method():
            log.error(f'MP start method {mp_start_method!r} is set for Flow, it should also be set'
                      f' in the if __name__ == "__main__" clause of the main module')
            raise MPStartMethodValueError(f'Multiprocessing start method mismatch: '
                                          f'got {mp.get_start_method()!r} for main process '
                                          f'and {mp_start_method!r} for Flow')
        self._mp_start_method = mp_start_method

        if not metrics_enabled:
            log.warn('Metrics collecting is disabled')
            metrics_collector = Collector(collectible_metrics=[])
        self._metrics_manager = get_metrics_manager(metrics_collector, metrics_exporter)

        # Parse steps: list[FlowStep] = parallel, single = sequential
        self._flow_steps: List[Union[FlowStep, List[FlowStep]]] = []
        for step in steps:
            if isinstance(step, list):
                flow_steps = [s if isinstance(s, FlowStep) else FlowStep(s) for s in step]
                self._flow_steps.append(flow_steps)
            elif isinstance(step, FlowStep):
                self._flow_steps.append(step)
            else:
                self._flow_steps.append(FlowStep(step))
        
        if not self._flow_steps:
            raise ValueError("Flow requires at least one step")

        # Simplified queue structure: List[List[List[FlowStepQueue]]]
        # Each step always has a list of queues (even if just one queue)
        self._queues: List[List[List[FlowStepQueue]]] = []

        # Add parallel tasks collector handler if last step is a list (parallel steps)
        if isinstance(self._flow_steps[-1], list):
            self._flow_steps.append(FlowStep(
                ParallelTasksCollectorHandler(),
            ))

    @property
    def state(self):
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == FlowState.RUNNING

    @cached_property
    def need_collect_task_timers(self) -> bool:
        return self._metrics_manager.collector.is_collectible(MetricsTypes.TASK_TIMERS)

    def start(self, timeout: Optional[int] = None):
        """
        Starts Flow and waits for all subprocesses to initialize.

        raising FlowError if 'timeout' is set and some handler was not able to initialize in time.
        """
        log.info('Flow is starting')
        self._state = FlowState.STARTING
        self._run_steps(timeout)
        self._run_tasks()
        self._state = FlowState.RUNNING
        log.info('Flow was started')

    async def process(self, task: BaseTask, timeout_sec: float = 5.) -> bool:
        """
        Starts processing task through pipeline and returns True if all was ok.
        The result is saved in the source task
        """
        if not self.is_running:
            raise NotRunningError

        with timeit() as timer:
            future = asyncio.Future()
            self._task_futures[task.task_id] = future

            task.priority = min(task.priority, self._queue_priorities - 1)
            task.set_timeout(timeout_sec)
            task.metrics.start_transfer_timer(MAIN_PROCESS)

            start_time = monotonic()
            
            # Handle task distribution - check if we have parallel flow structure
            if any(isinstance(step, list) for step in self._flow_steps):
                # New parallel flow structure
                async def distribute_task(task: BaseTask, timeout_sec: float, start_time: float):
                    """Task distribution using centralized queue logic with async retry for full queues."""
                    from .queues import find_target_queues
                    
                    # Find target queues starting from step 0 (first step)
                    target_queues = find_target_queues(
                        queues=self._queues,
                        task=task,
                        current_step=0
                    )
                    
                    # Send task to all target queues with retry logic for full queues
                    for to_queue in target_queues:
                        while self.state != FlowState.STOPPED and (monotonic() - start_time) < timeout_sec:
                            try:
                                to_queue.put(task, block=False)
                            except queue.Full:
                                await asyncio.sleep(0.001)
                            else:
                                break
                
                await distribute_task(task, timeout_sec, start_time)
            else:
                # Original sequential flow structure
                to_queue = select_next_queue(
                    queues=self._queues,
                    task=task,
                )
                while self.state != FlowState.STOPPED and (monotonic() - start_time) < timeout_sec:
                    try:
                        to_queue.put(task, block=False)
                    except queue.Full:
                        await asyncio.sleep(0.001)
                    else:
                        break

            elapsed_time = monotonic() - start_time

            tasks_stats = TasksStats()
            try:
                finished_task: BaseTask = await asyncio.wait_for(
                    future,
                    timeout=(timeout_sec - elapsed_time),
                )
            # todo is it correct to hide a specific error behind a general FlowError?
            except asyncio.TimeoutError:
                tasks_stats.timeout += 1
                raise FlowError('Task timeout error')
            except asyncio.CancelledError:
                tasks_stats.cancel += 1
                if self.state in (FlowState.STOPPING, FlowState.STOPPED):
                    raise FlowError('Task was cancelled')
                else:
                    # process was cancelled by external actor, so reraise
                    raise

            else:
                tasks_stats.complete += 1
            finally:
                self._metrics_manager.collector.add_tasks_stats(tasks_stats)
                del self._task_futures[task.task_id]

        finished_task.metrics.handle_times.add('total', timer.seconds)

        if self.need_collect_task_timers:
            self._metrics_manager.collector.add_task_metrics(finished_task.metrics)

        task.update(finished_task)

        return True

    async def stop(self, graceful: bool = True):
        if not self.is_running:
            log.info('Flow is not running')
            return

        self._state = FlowState.STOPPING
        log.info('Flow is stopping')

        if graceful:
            first_queue = self._queues[DEFAULT_PRIORITY][0][0].queue  # Always [step][queue_index]
            first_queue.put(StopTask())
            await asyncio.sleep(3)

        self._metrics_manager.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._task_futures.values():
            task.cancel()

        self._join_context(self._processes_context)

        self._state = FlowState.STOPPED
        log.info('Flow was stopped')

    def _calc_queue_size(self, step: Union[FlowStep, List[FlowStep]]):
        """ If queue size not specified manually, get queue size based on batch size for handler.
        We need at least batch_size places in queue and then some additional space
        """
        if self._queue_size is not None:
            return self._queue_size

        # queue should be able to store at least 20 task, that's seems reasonable

        if isinstance(step, list):
            return max(sum(s.batch_size*3 for s in step), 20)
        else:
            return max(step.batch_size*3, 20)

    async def _check_memory_usage(self, sleep_sec: float = 1.):
        handler_processes_dict = {}

        for step_number, step in enumerate(self._flow_steps, 1):
            if isinstance(step, list):
                for parallel_index, step_ in enumerate(step, 1):
                    name = step_.handler.get_step_name(step_number, parallel_index)
                    pids = self._contexts[step_.handler].pids()
                    processes = []
                    for pid in pids:
                        process = psutil.Process(pid)
                        processes.append(process)
                    handler_processes_dict[name] = processes
            else:
                name = step.handler.get_step_name(step_number, None)
                pids = self._contexts[step.handler].pids()
                processes = []
                for pid in pids:
                    process = psutil.Process(pid)
                    processes.append(process)
                handler_processes_dict[name] = processes

        while self.state != FlowState.STOPPED:
            metrics = MetricsItems()
            all_memory_usage = 0
            for flow_step_name, processes in handler_processes_dict.items():
                nprocs_memory_sum = 0
                for process in processes:
                    memory = process.memory_info().rss
                    nprocs_memory_sum  += memory
                    metrics.add(flow_step_name, memory)
                all_memory_usage += nprocs_memory_sum
                if len(processes) != 1:
                    metrics.add(f'{flow_step_name}_nprocs_sum', nprocs_memory_sum)
            metrics.add('all_memory_usage', all_memory_usage)
            self._metrics_manager.collector.add_memory_usage(metrics)
            await asyncio.sleep(sleep_sec)

    def _run_steps(self, timeout: Optional[int]):
        if len(self._flow_steps) == 0:
            log.info('Flow has zero steps -> do nothing')
            return

        # Count total processes
        total_procs = 1  # main process
        for step_or_group in self._flow_steps:
            if isinstance(step_or_group, list):
                total_procs += sum(s.nprocs for s in step_or_group)
            else:
                total_procs += step_or_group.nprocs
        
        start_barrier = Barrier(total_procs)

        # Create uniform queue structure - every step has a list of queues
        for _ in range(self._queue_priorities):
            queues = []
            
            for step_or_group in self._flow_steps:
                if isinstance(step_or_group, list):
                    # Parallel steps: create list with multiple input queues
                    parallel_queues = []
                    for step in step_or_group:
                        queue_size = self._calc_queue_size(step)
                        parallel_queues.append(FlowStepQueue(
                            queue=TaskMetricsQueue(queue_size),
                            handle_condition=step.handle_condition,
                        ))
                    queues.append(parallel_queues)  # List with multiple queues
                else:
                    # Sequential step: create list with single queue
                    step = step_or_group
                    queue_size = self._calc_queue_size(step)
                    queues.append([FlowStepQueue(  # List with single queue
                        queue=TaskMetricsQueue(queue_size),
                        handle_condition=step.handle_condition,
                    )])
            
            # Final output queue - also a list with single queue
            final_step = self._flow_steps[-1]
            queue_size = self._calc_queue_size(
                final_step
            )
            queues.append([FlowStepQueue(  # List with single queue
                queue=TaskMetricsQueue(queue_size),
                handle_condition=operator.truth,
            )])
            
            self._queues.append(queues)

        step_number = 1
        
        for step_or_group in self._flow_steps:
            if isinstance(step_or_group, list):
                # Create parallel workers
                for i, step in enumerate(step_or_group):
                    worker = Worker(
                        queues=self._queues,
                        task_handler=step.handler,
                        batch_size=step.batch_size,
                        batch_timeout=step.batch_timeout,
                        batch_lock=mp.RLock() if step.nprocs > 1 and step.batch_size > 1 else None,
                        read_lock=mp.RLock(),
                        step_number=step_number,
                        parallel_index=i,  # Each parallel worker gets its own queue index
                    )
                    
                    self._contexts[step.handler] = start_processes(
                        worker.loop, nprocs=step.nprocs, join=False, daemon=True,
                        start_method=self._mp_start_method, args=(start_barrier,),
                        on_start_wait=step.on_start_wait,
                    )
                    
                    log.info(f'Created parallel step {step.handler.__class__.__name__}')
                
                step_number += 1
                
            else:
                # Sequential step
                worker = Worker(
                    queues=self._queues,
                    task_handler=step_or_group.handler,
                    batch_size=step_or_group.batch_size,
                    batch_timeout=step_or_group.batch_timeout,
                    batch_lock=mp.RLock() if step_or_group.nprocs > 1 and step_or_group.batch_size > 1 else None,
                    read_lock=mp.RLock(),
                    step_number=step_number,
                    parallel_index=None,  # Sequential step
                )
                
                self._contexts[step_or_group.handler] = start_processes(
                    worker.loop, nprocs=step_or_group.nprocs, join=False, daemon=True,
                    start_method=self._mp_start_method, args=(start_barrier,),
                    on_start_wait=step_or_group.on_start_wait,
                )
                
                log.info(f'Created sequential step {step_or_group.handler.__class__.__name__}')
                step_number += 1

        for queue_list in self._queues:
            for step_queues in queue_list:
                for step_queue in step_queues:
                    step_queue.queue.cancel_join_thread()

        try:
            log.info(f'Waiting for all workers to startup for {timeout} seconds...')
            start_barrier.wait(timeout)
        except BrokenBarrierError:
            raise TimeoutError('Starting timeout expired')

    def _run_tasks(self):
        """Start common background tasks"""
        self._tasks.append(asyncio.ensure_future(self._fetch_processed()))
        self._tasks.append(asyncio.ensure_future(self._check_is_alive()))

        self._metrics_manager.start(queues_info=self._get_queues_info())
        self._tasks.append(asyncio.ensure_future(self._check_memory_usage()))

    def _get_queues_info(self) -> Dict[mp.Queue, str]:
        """Returns queues between Step handlers and its names."""

        def step_queues_info() -> List[Tuple[int, Tuple[str, str, FlowStepQueue]]]:
            for priority in range(self._queue_priorities):
                from_step = MAIN_PROCESS

                for (step_number, step), queues in zip(enumerate(self._flow_steps, 1), self._queues[priority]):
                    if isinstance(step, list):
                        for (parallel_index, step_), queue_ in zip(enumerate(step, 1), queues):
                            handler = step_.handler
                            to_step = handler.get_step_name(step_number, parallel_index)
                            yield (priority, (from_step, to_step, queue_))
                        from_step = f"pstep{step_number}"
                    else:
                        queue_ = queues[0]
                        to_step = step.handler.get_step_name(step_number, None)
                        yield (priority, (from_step, to_step, queue_))
                        from_step = to_step

                to_step = MAIN_PROCESS
                yield (priority, (from_step, to_step, self._queues[priority][-1][0]))

        result = {}
        for priority, (from_step_name, to_step_name, queue_) in step_queues_info():
            name = f'p_{priority}_from_{from_step_name}_to_{to_step_name}' if priority > 0 else f'from_{from_step_name}_to_{to_step_name}'
            result[queue_.queue] = name
        return result

    @staticmethod
    def _fetch_from_queue(out_queue: mp.Queue) -> Union[BaseTask, None]:
        try:
            task = out_queue.get(timeout=1.)
            return task
        except queue.Empty:
            return None

    def _read_from_queue(self, loop: asyncio.AbstractEventLoop, q: mp.Queue) -> None:
        while self.state != FlowState.STOPPED:
            task = self._fetch_from_queue(q)

            if task is None:
                continue

            fut = self._task_futures.get(task.task_id)
            if fut and not fut.cancelled() and not fut.done():
                task.metrics.stop_transfer_timer(MAIN_PROCESS, task.priority)
                task_size = getattr(q, 'task_size', None)
                if task_size:
                    task.metrics.save_task_size(task_size, MAIN_PROCESS, task.priority)

                loop.call_soon_threadsafe(fut.set_result, task)

    async def _fetch_processed(self):
        """Fetching messages from output queue.

        To handle messages from another process and not block asyncio loop, we run queue.get()
        in a separate thread

        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self._queue_priorities) as queue_fetch_executor:
            results_queues = [
                queues_with_same_priority[-1][0].queue for queues_with_same_priority in self._queues
            ]
            tasks = [
                loop.run_in_executor(queue_fetch_executor, self._read_from_queue, loop, q)
                for q in results_queues
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
            for t in done:
                try:
                    t.result()
                except Exception:
                    log.exception('error in queue-consumer')

            for t in pending:
                t.cancel()

    async def _check_is_alive(self, sleep_sec: float = 1.):
        """Checks that all child processes are alive.

        If at least one process is not alive, it stops Flow.
        """
        while self.state != FlowState.STOPPED:
            processes_stats = ProcessesStats()
            for handler, context in self._contexts.items():
                for proc in context.processes:
                    if not proc.is_alive():
                        if self.is_running:
                            handler_name = handler.__class__.__name__
                            log.error('The process %s for %s handler is dead',
                                      proc.pid, handler_name)
                            processes_stats.add_dead_process()
                            self._metrics_manager.collector.add_processes_stats(processes_stats)
                            await self.stop(graceful=False)
                    else:
                        processes_stats.add_running_process()
            self._metrics_manager.collector.add_processes_stats(processes_stats)
            await asyncio.sleep(sleep_sec)

    @staticmethod
    def _join_context(context: ProcessContext, timeout_sec: float = 0.01):
        try:
            context.join(timeout_sec)
        except (ProcessExitedException, ProcessRaisedException):
            pass

        for p in context.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)

    @property
    def _processes_context(self) -> ProcessContext:
        processes, error_queues = [], []
        for context in self._contexts.values():
            processes.extend(context.processes)
            error_queues.extend(context.error_queues)
        return ProcessContext(processes=processes, error_queues=error_queues)
