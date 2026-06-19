import multiprocessing as mp
import threading
from multiprocessing.queues import Queue


class TaskMetricsQueue(Queue):
    """A multiprocessing Queue that records the serialized size of the last
    received task.

    The size is captured at ``_recv_bytes`` time (the raw pickled byte length)
    and stored in thread-local storage so that it stays consistent even when
    several threads call ``get()`` on the same queue concurrently (which happens
    with the multi-threaded result drain on the main process). The previous
    implementation kept ``_task_size`` on the queue instance, which produced a
    data race between concurrent ``get`` calls.

    Backwards compatibility: the ``task_size`` property is preserved; it now
    returns the size observed by the *current* thread's most recent ``get``.
    """

    def __init__(self, maxsize=0, *, ctx=None):
        ctx = ctx or mp.get_context()
        Queue.__init__(self, maxsize, ctx=ctx)
        self._local = threading.local()

    def _recv_bytes_wrapper(self, *args, **kwargs):
        """Wraps _recv_bytes to record task byte size per-thread."""
        result = self.__recv_bytes(*args, **kwargs)
        self._local.task_size = len(result)
        return result

    @property
    def _recv_bytes(self):
        return self._recv_bytes_wrapper

    @_recv_bytes.setter
    def _recv_bytes(self, _recv_bytes_function):
        self.__recv_bytes = _recv_bytes_function

    @property
    def task_size(self) -> int:
        return getattr(self._local, 'task_size', 0)
