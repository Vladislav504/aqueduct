from typing import Dict, Optional

from .base import IExtendable, MetricsItems
from .timer import TransferTimer

_METRICS_ITEM_FIELDS = (
    'transfer_times',
    'task_sizes',
    'handle_times',
    'batch_times',
    'batch_sizes',
)


class TasksMetricsStorage(IExtendable):
    def __init__(self):
        self.transfer_times = MetricsItems()
        self.task_sizes = MetricsItems()
        self.handle_times = MetricsItems()
        self.batch_times = MetricsItems()
        self.batch_sizes = MetricsItems()
        # Set at fan-out time: per-list lengths before branch-local metrics begin.
        self._fanout_offsets: Optional[Dict[str, int]] = None

    def stamp_fanout_offsets(self):
        """Record current list lengths so parallel assembly can skip the shared prefix."""
        self._fanout_offsets = {
            name: len(getattr(self, name).items)
            for name in _METRICS_ITEM_FIELDS
        }

    def clear_fanout_offsets(self):
        self._fanout_offsets = None

    def extend(self, storage: 'TasksMetricsStorage'):
        self.transfer_times.extend(storage.transfer_times)
        self.task_sizes.extend(storage.task_sizes)
        self.handle_times.extend(storage.handle_times)
        self.batch_times.extend(storage.batch_times)
        self.batch_sizes.extend(storage.batch_sizes)

    def extend_parallel_branch(self, branch_metrics: 'TasksMetricsStorage'):
        """Merge only the branch-local tail of *branch_metrics* (post fan-out offset)."""
        if branch_metrics is None:
            return

        offsets = branch_metrics._fanout_offsets or self._fanout_offsets
        if not offsets:
            self.extend(branch_metrics)
            return

        for name in _METRICS_ITEM_FIELDS:
            offset = offsets.get(name, 0)
            items = getattr(branch_metrics, name).items
            if offset < len(items):
                getattr(self, name).add_items(items[offset:])


class TaskMetrics(TasksMetricsStorage):
    """Task's "backpack" with metrics.

    It is used to store metrics related to a task passing through the Flow child processes. This mechanic
    works as long as all tasks reach the output queue in the main process.
    """
    def __init__(self):
        super().__init__()
        self._transfer_timer: TransferTimer = None  # noqa

    def start_transfer_timer(self, transfer_from: str):
        self._transfer_timer = TransferTimer(transfer_from)
        self._transfer_timer.start()

    def stop_transfer_timer(self, transfer_to: str, priority: int = 0):
        self._transfer_timer.stop()
        from_ = self._transfer_timer.transfer_from
        name = (
            f'p_{priority}_from_{from_}_to_{transfer_to}'
            if priority > 0 else f'from_{from_}_to_{transfer_to}'
        )
        self.transfer_times.add(name, self._transfer_timer.seconds)

    def save_task_size(self, task_size: int, transfer_to: str, priority: int = 0):
        from_ = self._transfer_timer.transfer_from
        name = (
            f'p_{priority}_from_{from_}_to_{transfer_to}'
            if priority > 0 else f'from_{from_}_to_{transfer_to}'
        )
        self.task_sizes.add(name, task_size)
