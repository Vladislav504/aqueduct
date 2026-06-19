Parallel Flow
#############

Aqueduct lets a single step fan a task out to several handlers that run **in parallel**
(each in its own os process) and then collects their results back into the original task.
This is useful when a task needs several *independent* CPU-bound computations
(for example: several models scoring the same image), since the branches run
concurrently instead of one after another.


Defining a parallel step
************************
A parallel step is just a **list** of handlers (or ``FlowStep`` objects) passed to ``Flow``.
A single handler (not in a list) is a regular sequential step, exactly as before.

.. code-block:: python

    from aqueduct import Flow, FlowStep, BaseTaskHandler, BaseTask


    class Task(BaseTask):
        def __init__(self):
            super().__init__()
            self.score_a = None
            self.score_b = None


    class ModelA(BaseTaskHandler):
        def handle(self, *tasks: Task):
            for task in tasks:
                task.score_a = 1


    class ModelB(BaseTaskHandler):
        def handle(self, *tasks: Task):
            for task in tasks:
                task.score_b = 2


    # ModelA and ModelB process the same task in parallel.
    flow = Flow([ModelA(), ModelB()])
    flow.start()

    task = Task()
    await flow.process(task)
    assert task.score_a == 1
    assert task.score_b == 2

    await flow.stop()

You can mix sequential and parallel steps freely. A sequential step that immediately
follows a parallel step acts as the **collector** for that group:

.. code-block:: python

    flow = Flow(
        PreprocessHandler(),       # sequential step
        [ModelA(), ModelB()],      # parallel step (fan-out)
        AggregateHandler(),        # sequential step that sees the merged task
    )

If the parallel group is the last step, Aqueduct automatically appends an internal
collector, so you do not have to add one yourself.


How a task flows through a parallel step
****************************************
1. The task is sent (fanned out) to every branch whose ``handle_condition`` matches.
2. Each branch processes its **own copy** of the task in a separate process.
3. The branch results are collected and merged back into a single task.
4. The merged task continues to the next step (or is returned to ``process``).


Result collection and merge semantics
*************************************
Each branch receives an independent copy of the task and writes its own fields.
When all expected branches have returned, their copies are merged into one task.

- Branches are merged in **ascending branch order** (the order in which the handlers
  appear in the list).
- If two branches write the **same** field, the value from the **later** branch
  (higher index in the list) wins. Have branches write to distinct fields to avoid
  relying on this.
- Shared-memory fields (see `Shared memory <share_memory.rst>`_) are merged safely:
  a field that is already shared is not overwritten.

.. code-block:: python

    flow = Flow([ModelA(), ModelB()])  # ModelB wins on any field both write


Conditional branches
====================
Use ``handle_condition`` to send a task only to some branches. A task is counted as
"expected" by exactly the branches whose condition matched at fan-out time, so the
collector knows how many results to wait for.

.. code-block:: python

    flow = Flow([
        FlowStep(TypeAHandler(), handle_condition=lambda t: t.kind == 'a'),
        FlowStep(TypeBHandler(), handle_condition=lambda t: t.kind == 'b'),
    ])

A task that matches no branch condition is passed through to the next step unchanged.

.. note::
   The number of branches a task fans out to is fixed at the moment of fan-out.
   A branch handler may safely mutate the fields that other branches' conditions
   read — it will not change how many results the collector waits for.


Timeouts and expiration
=======================
``process`` accepts ``timeout_sec`` (5 seconds by default). If a branch fails or is
too slow and not all results arrive in time, the task expires: whatever partial
results were collected are assembled and the awaiting ``process`` call raises
``FlowError``. Abandoned partial collections are evicted automatically, so a failing
branch does not leak memory.


Constraints
===========
The collector keeps partial results in memory, keyed by task id, in a single process.
For this reason a step that collects a parallel group is always run with ``nprocs == 1``;
if you configure a higher value it is forced back to ``1`` (with a warning) to keep
result assembly correct.

In the example below ``AggregateHandler`` directly follows the parallel group, so it is
the collector. Even though it is configured with ``nprocs=4``, Aqueduct forces it to
``1`` and logs a warning:

.. code-block:: python

    collector_step = FlowStep(AggregateHandler(), nprocs=4)
    flow = Flow(
        [ModelA(), ModelB()],   # parallel group
        collector_step,         # collector -> forced to nprocs == 1
    )

    assert collector_step.nprocs == 1  # was 4, forced back to 1

The same applies to the internal collector that Aqueduct appends automatically when a
parallel group is the last step — it always runs in a single process.

The restriction only affects the **collecting** step. The parallel branches themselves,
and any non-collecting step, can still use ``nprocs > 1`` freely:

.. code-block:: python

    flow = Flow(
        FlowStep(PreprocessHandler(), nprocs=3),   # ok: not a collector
        [
            FlowStep(ModelA(), nprocs=2),          # ok: parallel branch
            FlowStep(ModelB(), nprocs=2),          # ok: parallel branch
        ],
        FlowStep(AggregateHandler(), nprocs=4),    # collector -> forced to 1
    )

If you need the collecting step to scale across processes, split it in two: keep a
lightweight single-process collector right after the parallel group, then forward to a
separate heavy step that can use ``nprocs > 1``:

.. code-block:: python

    flow = Flow(
        [ModelA(), ModelB()],                       # parallel group
        AggregateHandler(),                         # collector (nprocs == 1)
        FlowStep(HeavyPostprocessHandler(), nprocs=4),  # scales freely
    )


Zero-copy fan-out
*****************
When a task is fanned out to ``N`` branches, by default its payload is pickled and
copied ``N`` times (once per branch). For large payloads (images, numpy arrays, byte
buffers) this can dominate the cost of fan-out.

Zero-copy fan-out moves large payload fields into shared memory **once** and sends only
a lightweight handle to every branch, so the per-branch transfer cost becomes ``O(1)``
in payload size instead of ``O(N)``. It is controlled by ``Flow`` constructor arguments:

- ``zero_copy_fanout`` - enable zero-copy fan-out. Default is ``False``.
- ``fanout_min_share_bytes`` - minimum field size (in bytes) worth moving to shared
  memory. Smaller fields are copied as usual. Defaults to 64 KiB.

.. code-block:: python

    flow = Flow(
        PreprocessHandler(),
        [ModelA(), ModelB()],
        zero_copy_fanout=True,
    )

.. warning::
   With zero-copy fan-out enabled, all branches share the **same** underlying buffer
   for a moved field. Branches must treat such payload fields as **read-only** — an
   in-place modification in one branch would be visible to the others and cause a data
   race. If you need to mutate the payload, leave ``zero_copy_fanout`` disabled (the
   default), so each branch gets an independent copy.

Backwards compatibility: zero-copy is fully optional and falls back to the regular copy
path whenever a field cannot be shared (no shareable fields, ``numpy`` not installed,
payload below the threshold, etc.).


Metrics
*******
Metrics for a fanned-out task are aggregated correctly: work done before the fan-out
(shared by all branches) is counted once, and each branch's own segment is counted once.
See `Metrics <metrics.rst>`_ for how to export them.
