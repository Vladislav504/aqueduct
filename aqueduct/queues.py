import asyncio
import multiprocessing as mp
import queue
from dataclasses import dataclass
from time import monotonic
from typing import Callable, List, Tuple

from .handler import HandleConditionType
from .task import BaseTask


@dataclass
class FlowStepQueue:
    queue: mp.Queue
    handle_condition: HandleConditionType


QueueWithBranch = Tuple[mp.Queue, int]


def select_next_queue(queues: List[List[List[FlowStepQueue]]], task: BaseTask, start_index: int = 0) -> mp.Queue:
    targets = find_target_queues_with_branches(queues, task, start_index)
    return targets[0][0]


def find_target_queues_with_branches(
    queues: List[List[List[FlowStepQueue]]],
    task: BaseTask,
    current_step: int,
) -> List[QueueWithBranch]:
    """Find target queues and the parallel branch index for each match."""
    priority_queues = queues[task.priority]

    for step_index in range(current_step, len(priority_queues)):
        step_queues = priority_queues[step_index]

        if len(step_queues) > 1:
            target_queues: List[QueueWithBranch] = []
            for branch_index, queue_info in enumerate(step_queues):
                if queue_info.handle_condition(task):
                    target_queues.append((queue_info.queue, branch_index))

            if target_queues:
                return target_queues

        else:
            queue_info = step_queues[0]
            if queue_info.handle_condition(task):
                return [(queue_info.queue, 0)]

    final_queue = priority_queues[-1][0].queue
    return [(final_queue, 0)]


def find_target_queues(
    queues: List[List[List[FlowStepQueue]]],
    task: BaseTask,
    current_step: int,
) -> List[mp.Queue]:
    return [q for q, _ in find_target_queues_with_branches(queues, task, current_step)]


def _put_task_to_queues(
    task: BaseTask,
    targets: List[QueueWithBranch],
    *,
    zero_copy_fanout: bool = False,
    min_share_bytes: int = None,
    block: bool = True,
) -> None:
    """Send *task* to every target queue, stamping fan-out metadata when needed."""
    fanout_degree = len(targets)

    if fanout_degree > 1:
        # NOTE: the per-branch ``_fanout_branch_index`` is intentionally NOT set
        # here. ``mp.Queue.put`` pickles the task on a background feeder thread, so
        # mutating the shared task object between puts races with serialization.
        # The receiving parallel worker stamps its own branch index in
        # ``Worker._post_handle`` (from its ``parallel_index``), which is the value
        # the collector relies on for ordering/assembly.
        task._fanout_expected_count = fanout_degree
        if task.metrics is not None:
            task.metrics.stamp_fanout_offsets()

        if zero_copy_fanout:
            from .fanout import prepare_task_for_fanout, DEFAULT_MIN_SHARE_BYTES
            prepare_task_for_fanout(
                task,
                fanout_degree=fanout_degree,
                min_share_bytes=min_share_bytes if min_share_bytes is not None else DEFAULT_MIN_SHARE_BYTES,
            )

    for target_queue, _branch_index in targets:
        if block:
            target_queue.put(task)
        else:
            target_queue.put(task, block=False)


def distribute_task_to_next_step(
    queues: List[List[List[FlowStepQueue]]],
    task: BaseTask,
    current_step: int,
    zero_copy_fanout: bool = False,
    min_share_bytes: int = None,
) -> bool:
    """Distribute task to the next appropriate step queues (synchronous, for workers)."""
    targets = find_target_queues_with_branches(queues, task, current_step + 1)
    _put_task_to_queues(
        task,
        targets,
        zero_copy_fanout=zero_copy_fanout,
        min_share_bytes=min_share_bytes,
        block=True,
    )
    return True


async def distribute_task_async(
    queues: List[List[List[FlowStepQueue]]],
    task: BaseTask,
    current_step: int,
    should_continue: Callable[[], bool],
    zero_copy_fanout: bool = False,
    min_share_bytes: int = None,
) -> None:
    """Distribute task with async retry when a queue is full (for Flow.process)."""
    targets = find_target_queues_with_branches(queues, task, current_step)
    fanout_degree = len(targets)

    if fanout_degree > 1:
        # See ``_put_task_to_queues``: branch index is stamped by the receiving
        # worker, not here, to avoid racing with the queue's pickling thread.
        task._fanout_expected_count = fanout_degree
        if task.metrics is not None:
            task.metrics.stamp_fanout_offsets()

        if zero_copy_fanout:
            from .fanout import prepare_task_for_fanout, DEFAULT_MIN_SHARE_BYTES
            prepare_task_for_fanout(
                task,
                fanout_degree=fanout_degree,
                min_share_bytes=min_share_bytes if min_share_bytes is not None else DEFAULT_MIN_SHARE_BYTES,
            )

    for target_queue, _branch_index in targets:
        while should_continue():
            try:
                target_queue.put(task, block=False)
            except queue.Full:
                await asyncio.sleep(0.001)
            else:
                break
