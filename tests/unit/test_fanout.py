"""Tests for the zero-copy fan-out improvements and parallel-flow fixes.

These cover the MVP additions:
  A. zero-copy fan-out (aqueduct/fanout.py)
  B. collector correctness (single-proc enforcement, expired-path bug fix)
  C. batched/multi-threaded final drain configuration
  D. per-task size metric (thread-local)
  E. backwards compatibility of the public API
"""
import asyncio
import os

import numpy as np
import pytest

from aqueduct.fanout import (
    DEFAULT_MIN_SHARE_BYTES,
    estimate_fanout_payload_savings,
    prepare_task_for_fanout,
)
from aqueduct.flow import Flow, FlowStep
from aqueduct.handler import BaseTaskHandler
from aqueduct.parallel_tasks import ParallelTaskCollector, TaskCollectionInfo
from aqueduct.task import BaseTask
from tests.unit.conftest import Task, run_flow


# ---------------------------------------------------------------------------
# A. fanout helpers
# ---------------------------------------------------------------------------

class PayloadTask(BaseTask):
    def __init__(self, payload=None):
        super().__init__()
        if payload is not None:
            self.payload = payload


def test_prepare_task_for_fanout_shares_large_array():
    big = np.zeros(DEFAULT_MIN_SHARE_BYTES // 8 + 1024, dtype=np.float64)
    task = PayloadTask(big)

    shared = prepare_task_for_fanout(task, fanout_degree=3)

    assert shared is True
    assert 'payload' in task._shared_fields
    # data still accessible and equal
    assert np.array_equal(task.payload, big)


def test_prepare_task_for_fanout_skips_small_payload():
    small = np.zeros(8, dtype=np.float64)  # well below threshold
    task = PayloadTask(small)

    shared = prepare_task_for_fanout(task, fanout_degree=4)

    assert shared is False
    assert 'payload' not in task._shared_fields


def test_prepare_task_for_fanout_noop_for_single_branch():
    big = np.zeros(DEFAULT_MIN_SHARE_BYTES + 1, dtype=np.uint8)
    task = PayloadTask(big)

    # fanout_degree == 1 means no duplication, so nothing to share
    assert prepare_task_for_fanout(task, fanout_degree=1) is False
    assert 'payload' not in task._shared_fields


def test_prepare_task_for_fanout_keeps_already_shared_field():
    big = np.zeros(DEFAULT_MIN_SHARE_BYTES + 1, dtype=np.uint8)
    task = PayloadTask(big)
    task.share_value('payload')

    # Should not raise and should not double-share
    shared = prepare_task_for_fanout(task, fanout_degree=3)
    assert shared is False
    assert 'payload' in task._shared_fields


def test_prepare_task_for_fanout_handles_bytes():
    payload = b'x' * (DEFAULT_MIN_SHARE_BYTES + 10)
    task = PayloadTask(payload)

    shared = prepare_task_for_fanout(task, fanout_degree=2)
    assert shared is True
    assert task.payload == payload


def test_estimate_fanout_payload_savings():
    payload = np.zeros(1000, dtype=np.uint8)  # 1000 bytes
    task = PayloadTask(payload)
    # 3 branches -> 2 extra copies avoided -> 2 * 1000
    assert estimate_fanout_payload_savings(task, 3) == 2000
    assert estimate_fanout_payload_savings(task, 1) == 0


# ---------------------------------------------------------------------------
# B. collector correctness
# ---------------------------------------------------------------------------

def test_collection_info_received_alias_backwards_compatible():
    info = TaskCollectionInfo(2)
    t = Task()
    info.add_result(t)
    # both the corrected name and the old misspelled alias work
    assert info.received_results == [t]
    assert info.recieved_results == [t]


def test_get_expired_result_does_not_raise_typeerror():
    """Regression: previous code called dict.pop(key=..., default=...) which
    raises TypeError. The fixed version must return cleanly."""
    # Build a 2-step queue layout where step 2 collects from a 2-branch step 1.
    from aqueduct.queues import FlowStepQueue
    import operator

    class _DummyQ:
        def put(self, *a, **k):
            pass

    step1 = [
        FlowStepQueue(queue=_DummyQ(), handle_condition=operator.truth),
        FlowStepQueue(queue=_DummyQ(), handle_condition=operator.truth),
    ]
    step2 = [FlowStepQueue(queue=_DummyQ(), handle_condition=operator.truth)]
    queues = [[step1, step2]]  # one priority

    collector = ParallelTaskCollector(step_number=2, queues=queues)
    assert collector._collect_parallel_results is True

    task = Task()
    task._fanout_expected_count = 2
    task._fanout_branch_index = 0
    # feed one partial result so a collection entry exists
    partial = collector.get_result(task)
    assert partial is None
    # now request the expired assembly — must not raise
    result = collector.get_expired_result(task)
    assert result is not None
    assert result.task_id == task.task_id


def test_collector_nprocs_forced_to_one():
    """A sequential step collecting a parallel group must be forced to nprocs=1."""
    class Pre(BaseTaskHandler):
        def handle(self, *tasks):
            pass

    class A(BaseTaskHandler):
        def handle(self, *tasks):
            pass

    class B(BaseTaskHandler):
        def handle(self, *tasks):
            pass

    class Collector(BaseTaskHandler):
        def handle(self, *tasks):
            pass

    collector_step = FlowStep(Collector(), nprocs=4)
    flow = Flow(
        Pre(),
        [A(), B()],
        collector_step,
    )
    # collector_step follows the parallel group -> forced to 1
    assert collector_step.nprocs == 1


# ---------------------------------------------------------------------------
# C / E. integration: zero-copy fan-out end to end + backwards compat
# ---------------------------------------------------------------------------

class ArrayTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.image = np.arange(DEFAULT_MIN_SHARE_BYTES, dtype=np.uint8)
        # NOTE: result attributes (h1/h2) are intentionally NOT pre-declared.
        # The parallel collector merges branch results via BaseTask.update, which
        # only fills attributes that are *missing* on the base task. Pre-declaring
        # them would block the merge (this is the existing 1.14 merge semantics).


class BranchA(BaseTaskHandler):
    def handle(self, *tasks):
        for task in tasks:
            # read the (possibly shared) payload and record a checksum
            task.h1 = int(task.image.sum())


class BranchB(BaseTaskHandler):
    def handle(self, *tasks):
        for task in tasks:
            task.h2 = int(task.image[0])


class TestZeroCopyFlowIntegration:
    @pytest.fixture
    async def zero_copy_flow(self, loop):
        # Zero-copy is applied on worker->parallel fan-out (after Pre), not on
        # main->parallel first step (SHM + collector assembly can crash on macOS).
        flow = Flow(PreHandler(), [BranchA(), BranchB()], zero_copy_fanout=True)
        async with run_flow(flow) as f:
            yield f

    @pytest.fixture
    async def legacy_flow(self, loop):
        # zero_copy disabled -> old copy-per-branch behaviour
        flow = Flow(PreHandler(), [BranchA(), BranchB()], zero_copy_fanout=False)
        async with run_flow(flow) as f:
            yield f

    async def test_zero_copy_fanout_correct_results(self, zero_copy_flow):
        task = ArrayTask()
        expected_sum = int(task.image.sum())
        await zero_copy_flow.process(task)
        assert task.pre_done is True
        assert task.h1 == expected_sum
        assert task.h2 == 0

    async def test_legacy_fanout_correct_results(self, legacy_flow):
        task = ArrayTask()
        expected_sum = int(task.image.sum())
        await legacy_flow.process(task)
        assert task.pre_done is True
        assert task.h1 == expected_sum
        assert task.h2 == 0

    async def test_multiple_tasks_zero_copy(self, zero_copy_flow):
        tasks = [ArrayTask() for _ in range(5)]
        expected = [int(t.image.sum()) for t in tasks]
        await asyncio.gather(*[zero_copy_flow.process(t) for t in tasks])
        for t, exp in zip(tasks, expected):
            assert t.h1 == exp
            assert t.h2 == 0


class TestResultFetchThreads:
    async def test_configurable_fetch_threads(self, loop):
        flow = Flow([BranchA(), BranchB()], result_fetch_threads=4)
        async with run_flow(flow) as f:
            tasks = [ArrayTask() for _ in range(10)]
            await asyncio.gather(*[f.process(t) for t in tasks])
            for t in tasks:
                assert t.h1 is not None
                assert t.h2 == 0


# ---------------------------------------------------------------------------
# Regression tests for plan fixes (B4–B10)
# ---------------------------------------------------------------------------

def test_zero_copy_fanout_defaults_off():
    flow = Flow([BranchA(), BranchB()])
    assert flow._zero_copy_fanout is False


def test_metrics_stamp_and_extend_parallel_branch():
    from aqueduct.metrics.task import TaskMetrics

    base = TaskMetrics()
    base.handle_times.add('pre_step', 0.1)
    base.transfer_times.add('from_main_to_pre', 0.01)
    base.stamp_fanout_offsets()

    branch = TaskMetrics()
    branch.handle_times.add('pre_step', 0.1)
    branch.handle_times.add('branch_step', 0.2)
    branch.transfer_times.add('from_main_to_pre', 0.01)
    branch.transfer_times.add('from_pre_to_branch', 0.02)
    branch._fanout_offsets = dict(base._fanout_offsets)

    base.extend_parallel_branch(branch)

    assert base.handle_times.items == [('pre_step', 0.1), ('branch_step', 0.2)]
    assert base.transfer_times.items == [
        ('from_main_to_pre', 0.01),
        ('from_pre_to_branch', 0.02),
    ]


def test_collector_evicts_expired_partial_collections():
    import operator
    import time
    from aqueduct.queues import FlowStepQueue

    class _DummyQ:
        def put(self, *a, **k):
            pass

    step1 = [
        FlowStepQueue(queue=_DummyQ(), handle_condition=operator.truth),
        FlowStepQueue(queue=_DummyQ(), handle_condition=operator.truth),
    ]
    step2 = [FlowStepQueue(queue=_DummyQ(), handle_condition=operator.truth)]
    queues = [[step1, step2]]

    collector = ParallelTaskCollector(step_number=2, queues=queues)
    task = Task()
    task._fanout_expected_count = 2
    task._fanout_branch_index = 0
    task.set_timeout(0.01)

    assert collector.get_result(task) is None
    assert task.task_id in collector._task_collections

    time.sleep(0.02)
    collector._evict_expired_collections()
    assert task.task_id not in collector._task_collections


class PreHandler(BaseTaskHandler):
    def handle(self, *tasks):
        for task in tasks:
            task.pre_done = True


class CheckSharedBranch(BaseTaskHandler):
    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def handle(self, *tasks):
        for task in tasks:
            setattr(task, self.attr_name, 'image' in task._shared_fields)


class MutatingBranchA(BaseTaskHandler):
    def handle(self, *tasks):
        for task in tasks:
            task.task_type = 'mutated'
            task.a_done = True


class BranchBOnly(BaseTaskHandler):
    def handle(self, *tasks):
        for task in tasks:
            task.b_done = True


class OverwriteBranch(BaseTaskHandler):
    def __init__(self, value: str):
        self.value = value

    def handle(self, *tasks):
        for task in tasks:
            task.shared_field = self.value


class TestWorkerFanoutPath:
    """B8: zero-copy must run on worker->parallel fan-out, not only main->parallel."""

    async def test_worker_fanout_zero_copy_results(self, loop):
        flow = Flow(PreHandler(), [BranchA(), BranchB()], zero_copy_fanout=True)
        async with run_flow(flow) as f:
            task = ArrayTask()
            expected_sum = int(task.image.sum())
            await f.process(task)
            assert task.pre_done is True
            assert task.h1 == expected_sum
            assert task.h2 == 0

    async def test_worker_fanout_zero_copy_shares_payload(self, loop):
        """Verify zero-copy runs on worker->parallel fan-out (unit-level, no full flow)."""
        import operator
        from unittest.mock import MagicMock

        from aqueduct.queues import FlowStepQueue, distribute_task_to_next_step

        step_pre = [FlowStepQueue(queue=MagicMock(), handle_condition=operator.truth)]
        step_parallel = [
            FlowStepQueue(queue=MagicMock(), handle_condition=operator.truth),
            FlowStepQueue(queue=MagicMock(), handle_condition=operator.truth),
        ]
        step_final = [FlowStepQueue(queue=MagicMock(), handle_condition=operator.truth)]
        queues = [[step_pre, step_parallel, step_final]]

        task = ArrayTask()
        distribute_task_to_next_step(
            queues, task, current_step=0, zero_copy_fanout=True,
        )
        assert 'image' in task._shared_fields
        assert step_parallel[0].queue.put.called
        assert step_parallel[1].queue.put.called

    async def test_worker_fanout_legacy_copy(self, loop):
        flow = Flow(PreHandler(), [CheckSharedBranch('a_shared'), CheckSharedBranch('b_shared')],
                     zero_copy_fanout=False)
        async with run_flow(flow) as f:
            task = ArrayTask()
            await f.process(task)
            assert task.a_shared is False
            assert task.b_shared is False


class TestConditionMutationNoHang:
    """B4: fan-out degree is stamped at distribution, not re-evaluated after mutation."""

    async def test_branch_mutation_does_not_break_collection(self, loop):
        flow = Flow([
            FlowStep(MutatingBranchA(), handle_condition=lambda t: True),
            FlowStep(BranchBOnly(), handle_condition=lambda t: getattr(t, 'task_type', None) != 'mutated'),
        ])
        async with run_flow(flow) as f:
            task = Task()
            await f.process(task)
            assert task.a_done is True
            assert task.b_done is True


class TestDeterministicAssembly:
    """B10: overlapping fields follow deterministic branch-index order."""

    async def test_higher_branch_index_wins_on_conflict(self, loop):
        flow = Flow([OverwriteBranch('A'), OverwriteBranch('B')])
        async with run_flow(flow) as f:
            for _ in range(5):
                task = Task()
                await f.process(task)
                assert task.shared_field == 'B'


class TestParallelMetricsCorrectness:
    """B9: pre-fan-out metrics counted once, one entry per branch."""

    async def test_assembled_metrics_not_doubled(self, loop):
        from aqueduct.metrics.base import MetricsTypes
        from aqueduct.metrics.collect import Collector

        collector = Collector(collectible_metrics=[MetricsTypes.TASK_TIMERS])
        flow = Flow(PreHandler(), [BranchA(), BranchB()], metrics_collector=collector)
        flow.start()
        try:
            task = ArrayTask()
            await flow.process(task)
            storage = collector.extract_metrics()
        finally:
            await flow.stop(graceful=False)

        handle_names = [name for name, _ in storage.handle_times.items]
        assert handle_names.count('step1_PreHandler') == 1
        assert handle_names.count('step2_BranchA_pstep1') == 1
        assert handle_names.count('step2_BranchB_pstep2') == 1


class TestStopTaskBroadcast:
    """B7: graceful stop reaches every parallel branch."""

    async def test_graceful_stop_parallel_group(self, loop):
        flow = Flow([BranchA(), BranchB()])
        async with run_flow(flow) as f:
            assert f.state.name == 'RUNNING'
            await f.stop(graceful=True)
            assert f.state.name == 'STOPPED'

