
from time import monotonic
from typing import Dict, List, Optional, Set

from .queues import FlowStepQueue
from .task import BaseTask


class TaskCollectionInfo:
    """Information for tracking parallel task collection."""

    def __init__(self, expected_count: int, expiration_time: Optional[float] = None):
        self.received_results: List[BaseTask] = []
        self.received_branches: Set[int] = set()
        self.expected_count: int = expected_count
        self.expiration_time: Optional[float] = expiration_time

    # Backwards-compatible alias for the previously (mis)spelled attribute.
    @property
    def recieved_results(self) -> List[BaseTask]:  # noqa: D401 - kept for compatibility
        return self.received_results

    def add_result(self, task: BaseTask):
        """Add a result to the collection."""
        self.received_results.append(task)
        branch_index = getattr(task, '_fanout_branch_index', None)
        if branch_index is None:
            branch_index = len(self.received_branches)
        self.received_branches.add(branch_index)

    def is_complete(self) -> bool:
        """Check if we have all expected results."""
        return len(self.received_branches) >= self.expected_count


class ParallelTaskCollector:
    """Handles collection and assembly of parallel task results.

    NOTE on multiprocessing safety:
        The collector keeps the partial-results state in-process. If the step that
        consumes a parallel group runs with ``nprocs > 1``, branch results for the
        same ``task_id`` may be read by different collector processes, which would
        prevent ``is_complete()`` from ever firing. The Flow enforces ``nprocs == 1``
        for any step that collects a parallel group (see ``Flow._validate_steps``),
        so this state is always owned by a single process for a given task.
    """

    def __init__(self, step_number: int, queues: List[List[List[FlowStepQueue]]]):
        self._task_collections: Dict[str, TaskCollectionInfo] = {}
        self._collect_parallel_results = False

        if step_number >= 2:
            prev_step_index = step_number - 2
            if prev_step_index < len(queues[0]):
                prev_step_queues = queues[0][prev_step_index]
                if len(prev_step_queues) > 1:
                    self._collect_parallel_results = True

    def _evict_expired_collections(self) -> None:
        """Drop partial collections whose tasks have passed their expiration time."""
        now = monotonic()
        expired_ids = [
            task_id
            for task_id, info in self._task_collections.items()
            if info.expiration_time is not None and now >= info.expiration_time
        ]
        for task_id in expired_ids:
            del self._task_collections[task_id]

    def _expected_count(self, task: BaseTask) -> int:
        """Read fan-out degree stamped at distribution time (do not re-evaluate conditions)."""
        stamped = getattr(task, '_fanout_expected_count', None)
        if stamped is not None:
            return max(int(stamped), 1)
        return 1

    def get_expired_result(self, task: BaseTask) -> Optional[BaseTask]:
        """Check for expired parallel tasks and assemble whatever partial results we have."""
        if not self._collect_parallel_results:
            return None

        self._evict_expired_collections()

        collection_info: Optional[TaskCollectionInfo] = self._task_collections.pop(task.task_id, None)
        if not collection_info:
            return None

        if collection_info.received_results:
            assembled_task = self._assemble_results(collection_info.received_results)
            if assembled_task:
                return assembled_task

        return None

    def get_result(self, task: BaseTask) -> Optional[BaseTask]:
        """Collect parallel results and complete when all expected results are received."""
        if not self._collect_parallel_results:
            return task

        self._evict_expired_collections()

        task_id = task.task_id

        if task_id not in self._task_collections:
            expected_count = self._expected_count(task)
            self._task_collections[task_id] = TaskCollectionInfo(
                expected_count,
                expiration_time=task.expiration_time,
            )

        collection_info = self._task_collections[task_id]
        collection_info.add_result(task)

        if collection_info.is_complete():
            final_task = self._assemble_results(collection_info.received_results)
            del self._task_collections[task_id]
            return final_task

        return None

    def _assemble_results(self, results: List[BaseTask]) -> Optional[BaseTask]:
        """Assemble results from parallel branches into a single task.

        Branches are merged in ascending ``_fanout_branch_index`` order so that
        overlapping regular fields follow deterministic last-branch-wins
        semantics. Shared-memory fields are merged via the dedicated shared-field
        mechanism (only missing shared fields are attached) so we never replace a
        shared field with a raw view into another process' shared buffer, which
        would corrupt refcounting and can crash the collector when re-pickled.
        Metrics from non-base branches contribute only their post-fan-out tail.
        """
        if not results:
            return None

        sorted_results = sorted(
            results,
            key=lambda t: getattr(t, '_fanout_branch_index', 0),
        )

        base_immutable = set(BaseTask.__dict__.keys()) | {
            '_shared_fields', '_fanout_expected_count', '_fanout_branch_index',
        }

        final_task = sorted_results[0]
        for result in sorted_results[1:]:
            # Attach any shared fields that the base task is missing. This keeps
            # shared-memory fields as proper shared fields (excluded from pickling)
            # instead of degrading them to dangling buffer views.
            shared_field_names = final_task._update_shared_fields(result)

            for field, value in result.__dict__.items():
                if field in base_immutable or field in shared_field_names:
                    continue
                # Never overwrite a shared field with a plain (possibly aliased) value.
                if field in final_task._shared_fields or field in result._shared_fields:
                    continue
                setattr(final_task, field, value)

            if final_task.metrics and result.metrics:
                final_task.metrics.extend_parallel_branch(result.metrics)

        if final_task.metrics:
            final_task.metrics.clear_fanout_offsets()

        return final_task
