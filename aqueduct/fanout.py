"""Zero-copy fan-out helpers for the parallel flow.

The parallel flow sends *the same* task to several branch queues. With the plain
``Queue.put`` path every branch pickles the whole task independently, so a task
carrying a large payload (image / numpy array / tensor) is serialized and copied
through the OS pipe N times. On GPU hardware this also multiplies host<->device
pressure.

This module makes the fan-out copy the heavy payload into shared memory exactly
**once** and send only a lightweight handle to every branch. Each branch attaches
to the same shared buffer (reference counted via :class:`SharedMemoryWrapper`),
so the per-branch transfer cost becomes O(1) in payload size instead of O(N).

Design goals:
  * Fully backwards compatible: if a task cannot be shared (no shareable fields,
    numpy missing, etc.) we transparently fall back to the regular ``put``.
  * Opt-in: controlled by ``Flow(zero_copy_fanout=True)``. When enabled, branches
    must treat shared payload fields as read-only to avoid cross-branch races.
  * Correct refcounting: ``SharedMemoryWrapper.__getstate__`` already increments
    the atomic refcount on each pickle, so attaching from N branches and the
    eventual N ``__del__`` calls balance out.
"""

from typing import Any, Dict, List, Optional

from .logger import log
from .task import BaseTask

try:  # numpy is an optional dependency
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only when numpy missing
    _np = None


# Minimum payload size (bytes) worth moving to shared memory. Below this the
# SHM allocation + bookkeeping costs more than just pickling the bytes.
DEFAULT_MIN_SHARE_BYTES = 64 * 1024  # 64 KiB


def _estimate_nbytes(value: Any) -> int:
    """Best-effort size estimate for deciding whether sharing is worth it."""
    if _np is not None and isinstance(value, _np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    return 0


def _is_shareable(value: Any) -> bool:
    if _np is not None and isinstance(value, _np.ndarray):
        return True
    return isinstance(value, (bytes, bytearray))


def prepare_task_for_fanout(
    task: BaseTask,
    fanout_degree: int,
    min_share_bytes: int = DEFAULT_MIN_SHARE_BYTES,
) -> bool:
    """Move large, not-yet-shared fields of ``task`` into shared memory in place.

    This is only beneficial when the task is about to be sent to more than one
    branch (``fanout_degree > 1``). Returns ``True`` if at least one field was
    moved to shared memory.

    The operation reuses :meth:`SharedFieldsMixin.share_value`, so the fields are
    excluded from pickling and instead transferred as cheap shared-memory handles.
    Fields already shared by the user are left untouched.
    """
    if fanout_degree <= 1:
        return False
    if not isinstance(task, BaseTask):
        return False

    shared_any = False
    # Snapshot keys to avoid mutating the dict while iterating.
    for field_name in list(task.__dict__.keys()):
        if field_name.startswith('_'):
            continue
        if field_name in ('task_id', 'priority', 'expiration_time', 'metrics'):
            continue
        # Already shared?
        shared_fields = getattr(task, '_shared_fields', {})
        if field_name in shared_fields:
            continue

        value = task.__dict__.get(field_name)
        if not _is_shareable(value):
            continue
        if _estimate_nbytes(value) < min_share_bytes:
            continue

        try:
            task.share_value(field_name)
            shared_any = True
        except Exception as exc:  # pragma: no cover - defensive, fall back to copy
            log.debug('zero-copy fan-out: could not share field %r: %s', field_name, exc)

    return shared_any


def estimate_fanout_payload_savings(task: BaseTask, fanout_degree: int) -> int:
    """Returns the approximate number of bytes avoided by sharing instead of copying.

    Used by metrics/benchmarks. ``(N - 1) * payload`` bytes are saved because the
    payload is transferred once instead of N times.
    """
    if fanout_degree <= 1:
        return 0
    total = 0
    for field_name, value in task.__dict__.items():
        if field_name.startswith('_'):
            continue
        total += _estimate_nbytes(value)
    return total * (fanout_degree - 1)
