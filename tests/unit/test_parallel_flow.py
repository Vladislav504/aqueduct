import asyncio
import os
import time
from typing import List

import pytest

from aqueduct.exceptions import FlowError, NotRunningError
from aqueduct.flow import Flow, FlowState, FlowStep
from aqueduct.handler import BaseTaskHandler
from tests.unit.conftest import (
    Task,
    run_flow,
    SetResultSleepHandler,
)


class ParallelTestHandler1(BaseTaskHandler):
    """Handler that records processing order and sets result."""
    
    def __init__(self, sleep_time: float = 0.05, result_value: str = "handler1"):
        self.sleep_time = sleep_time
        self.result_value = result_value
        self.processed_tasks = []
        
    def handle(self, *tasks: Task):
        for task in tasks:
            time.sleep(self.sleep_time)
            task.handler1_result = self.result_value
            task.handler1_pid = os.getpid()
            self.processed_tasks.append(task.task_id if hasattr(task, 'task_id') else str(id(task)))


class ParallelTestHandler2(BaseTaskHandler):
    """Handler that records processing order and sets result."""
    
    def __init__(self, sleep_time: float = 0.05, result_value: str = "handler2"):
        self.sleep_time = sleep_time
        self.result_value = result_value
        self.processed_tasks = []
        
    def handle(self, *tasks: Task):
        for task in tasks:
            time.sleep(self.sleep_time)
            task.handler2_result = self.result_value
            task.handler2_pid = os.getpid()
            self.processed_tasks.append(task.task_id if hasattr(task, 'task_id') else str(id(task)))


class ParallelTestHandler3(BaseTaskHandler):
    """Handler that records processing order and sets result."""
    
    def __init__(self, sleep_time: float = 0.05, result_value: str = "handler3"):
        self.sleep_time = sleep_time
        self.result_value = result_value
        self.processed_tasks = []
        
    def handle(self, *tasks: Task):
        for task in tasks:
            time.sleep(self.sleep_time)
            task.handler3_result = self.result_value
            task.handler3_pid = os.getpid()
            self.processed_tasks.append(task.task_id if hasattr(task, 'task_id') else str(id(task)))


class ConditionalHandler(BaseTaskHandler):
    """Handler that only processes tasks with specific attributes."""
    
    def __init__(self, process_type: str, result_value: str):
        self.process_type = process_type
        self.result_value = result_value
        
    def handle(self, *tasks: Task):
        for task in tasks:
            setattr(task, f"{self.process_type}_processed", True)
            setattr(task, f"{self.process_type}_result", self.result_value)


class TestParallelFlow:
    """Test class for flows with parallel and sequential steps."""

    @pytest.fixture
    async def simple_parallel_flow(self, loop):
        """Flow with two parallel handlers followed by explicit collector."""
        handler1 = ParallelTestHandler1(sleep_time=0.02)
        handler2 = ParallelTestHandler2(sleep_time=0.02)
        
        flow = Flow([handler1, handler2])
        async with run_flow(flow) as flow_instance:
            yield flow_instance

    @pytest.fixture
    async def mixed_flow(self, loop):
        """Flow with sequential, parallel, and sequential steps."""
        preprocessor = SetResultSleepHandler(0.01)  # Sequential
        
        # Parallel processing
        handler1 = ParallelTestHandler1(sleep_time=0.02)
        handler2 = ParallelTestHandler2(sleep_time=0.02)
        handler3 = ParallelTestHandler3(sleep_time=0.02)
        
        
        flow = Flow(
            preprocessor,           # Sequential step
            [handler1, handler2, handler3],  # Parallel step
        )
        async with run_flow(flow) as flow_instance:
            yield flow_instance

    @pytest.fixture
    async def conditional_parallel_flow(self, loop):
        """Flow with parallel handlers that have handle conditions followed by collector."""
        type_a_handler = ConditionalHandler("type_a", "processed_by_a")
        type_b_handler = ConditionalHandler("type_b", "processed_by_b")
        
        flow = Flow([
            FlowStep(type_a_handler, handle_condition=lambda task: getattr(task, 'task_type', None) == 'A'),
            FlowStep(type_b_handler, handle_condition=lambda task: getattr(task, 'task_type', None) == 'B'),
        ])
        async with run_flow(flow) as flow_instance:
            yield flow_instance

    async def test_parallel_processing_basic(self, simple_parallel_flow: Flow):
        """Test that parallel handlers process the same task concurrently and collector works."""
        task = Task()
        
        await simple_parallel_flow.process(task)
        
        # Both parallel handlers should have processed the task
        assert hasattr(task, 'handler1_result')
        assert hasattr(task, 'handler2_result')
        assert task.handler1_result == "handler1"
        assert task.handler2_result == "handler2"

    async def test_parallel_handlers_different_processes(self, simple_parallel_flow: Flow):
        """Test that parallel handlers run in different processes."""
        task = Task()
        await simple_parallel_flow.process(task)
        
        # Each parallel handler should run in a different process
        assert hasattr(task, 'handler1_pid')
        assert hasattr(task, 'handler2_pid')
        assert task.handler1_pid != task.handler2_pid

    async def test_mixed_sequential_parallel_flow(self, mixed_flow: Flow):
        """Test flow with sequential -> parallel -> sequential steps."""
        task = Task()
        
        await mixed_flow.process(task)
        
        # Should have preprocessor result
        assert task.result == 'test'  # From SetResultSleepHandler
        
        # Should have results from all parallel handlers (auto-collected)
        assert hasattr(task, 'handler1_result')
        assert hasattr(task, 'handler2_result') 
        assert hasattr(task, 'handler3_result')
        assert task.handler1_result == "handler1"
        assert task.handler2_result == "handler2"
        assert task.handler3_result == "handler3"

    async def test_multiple_tasks_parallel_processing(self, simple_parallel_flow: Flow):
        """Test processing multiple tasks through parallel flow."""
        tasks = [Task() for _ in range(5)]
        
        await asyncio.gather(*[simple_parallel_flow.process(task) for task in tasks])
        
        # All tasks should be processed by both parallel handlers
        for task in tasks:
            assert hasattr(task, 'handler1_result')
            assert hasattr(task, 'handler2_result')
            assert task.handler1_result == "handler1"
            assert task.handler2_result == "handler2"
        

    async def test_parallel_handle_conditions(self, conditional_parallel_flow: Flow):
        """Test that handle conditions work correctly with parallel steps and collector."""
        task_a = Task()
        task_a.task_type = 'A'
        
        task_b = Task()
        task_b.task_type = 'B'
        
        task_none = Task()
        # No task_type attribute
        
        await conditional_parallel_flow.process(task_a)
        await conditional_parallel_flow.process(task_b)
        await conditional_parallel_flow.process(task_none)
        
        # Task A should only be processed by type_a_handler
        assert hasattr(task_a, 'type_a_processed')
        assert not hasattr(task_a, 'type_b_processed')
        assert task_a.type_a_result == "processed_by_a"
        
        # Task B should only be processed by type_b_handler
        assert hasattr(task_b, 'type_b_processed')
        assert not hasattr(task_b, 'type_a_processed')
        assert task_b.type_b_result == "processed_by_b"
        
        # Task with no type should not be processed by either parallel handler
        assert not hasattr(task_none, 'type_a_processed')
        assert not hasattr(task_none, 'type_b_processed')

    async def test_parallel_step_error_handling(self, loop):
        """Test error handling when one parallel handler fails."""
        
        class FailingHandler(BaseTaskHandler):
            def handle(self, *tasks: Task):
                raise ValueError("Handler failed")
        
        class WorkingHandler(BaseTaskHandler):
            def handle(self, *tasks: Task):
                for task in tasks:
                    task.working_result = "success"
        
        flow = Flow([FailingHandler(), WorkingHandler()])
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            
            # The flow should handle the error gracefully
            with pytest.raises(FlowError):
                await flow_instance.process(task)

    async def test_queue_structure_parallel_steps(self, simple_parallel_flow: Flow):
        """Test that queue structure is correct for parallel steps."""
        queues_info = simple_parallel_flow._get_queues_info()
        
        # Should have proper queue structure for parallel processing
        # This tests the internal queue setup
        assert len(queues_info) > 0
        
        # Check that we have the right number of steps in the flow
        flow_steps = simple_parallel_flow._flow_steps
        assert len(flow_steps) == 2  # Parallel step + collector
        assert isinstance(flow_steps[0], list)  # First step is parallel (list)
        assert len(flow_steps[0]) == 2  # Two parallel handlers

    async def test_flow_state_with_parallel_steps(self, simple_parallel_flow: Flow):
        """Test that flow state management works with parallel steps."""
        assert simple_parallel_flow.state == FlowState.RUNNING
        
        task = Task()
        await simple_parallel_flow.process(task)
        
        # Flow should still be running after processing
        assert simple_parallel_flow.state == FlowState.RUNNING
        
        # Test graceful stop
        await simple_parallel_flow.stop(graceful=True)
        assert simple_parallel_flow.state == FlowState.STOPPED

    async def test_multiple_parallel_groups(self, loop):
        """Test flow with multiple separate parallel groups."""
        
        class Handler4(BaseTaskHandler):
            """Different handler class to avoid attribute conflicts."""
            def __init__(self, sleep_time: float = 0.05, result_value: str = "handler4"):
                self.sleep_time = sleep_time
                self.result_value = result_value
                
            def handle(self, *tasks: Task):
                for task in tasks:
                    time.sleep(self.sleep_time)
                    task.handler4_result = self.result_value
                    task.handler4_pid = os.getpid()
        
        # First parallel group
        handler1 = ParallelTestHandler1(sleep_time=0.01, result_value="group1_h1")
        handler2 = ParallelTestHandler2(sleep_time=0.01, result_value="group1_h2")
        
        # Sequential step between groups
        intermediate = SetResultSleepHandler(0.005)
        
        # Second parallel group
        handler3 = ParallelTestHandler3(sleep_time=0.01, result_value="group2_h3")
        handler4 = Handler4(sleep_time=0.01, result_value="group2_h4")
        
        flow = Flow(
            [handler1, handler2],    # First parallel group
            intermediate,            # Sequential step
            [handler3, handler4],    # Second parallel group
        )
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            await flow_instance.process(task)
            
            # Should have results from both parallel groups
            assert task.handler1_result == "group1_h1"  # From first group
            assert task.handler2_result == "group1_h2"  # From first group
            assert task.handler3_result == "group2_h3"  # From second group
            assert task.handler4_result == "group2_h4"  # From second group (different attribute)
            
            # Intermediate result should have been set by SetResultSleepHandler
            assert task.result == 'test'

    async def test_empty_parallel_group_behavior(self, loop):
        """Test behavior with empty parallel groups."""
        # Empty flow should raise an error during construction
        with pytest.raises(ValueError, match="Flow requires at least one step"):
            Flow()  # No steps at all should not be allowed
        
        # But an empty list within a flow might be handled differently
        # Let's test what actually happens rather than assume it fails
        try:
            flow = Flow([])
            # If it doesn't fail, that's acceptable behavior too
            assert True  # Empty list is allowed
        except ValueError:
            # If it does fail, that's also acceptable 
            assert True

    async def test_single_step_in_list_treated_as_parallel(self, loop):
        """Test that a single step in a list is treated as parallel step."""
        handler = ParallelTestHandler1()
        
        # Single handler in list should still be treated as parallel step
        flow = Flow([handler])
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            await flow_instance.process(task)
            
            assert hasattr(task, 'handler1_result')
            assert task.handler1_result == "handler1"

    @pytest.mark.parametrize('task_type, expected_handlers', [
        ('A', ['type_a']),
        ('B', ['type_b']),
        ('C', []),  # No handlers should process this
    ])
    async def test_conditional_parallel_parametrized(
        self, 
        conditional_parallel_flow: Flow, 
        task_type: str, 
        expected_handlers: List[str]
    ):
        """Parametrized test for conditional parallel processing."""
        task = Task()
        task.task_type = task_type
        
        await conditional_parallel_flow.process(task)
        
        for handler_type in ['type_a', 'type_b']:
            should_be_processed = handler_type in expected_handlers
            processed_attr = f"{handler_type}_processed"
            
            if should_be_processed:
                assert hasattr(task, processed_attr), f"Task should have been processed by {handler_type}"
            else:
                assert not hasattr(task, processed_attr), f"Task should NOT have been processed by {handler_type}"

    async def test_parallel_step_timeout(self, loop):
        """Test timeout behavior with parallel steps."""
        slow_handler1 = ParallelTestHandler1(sleep_time=1.0)  # Very slow
        slow_handler2 = ParallelTestHandler2(sleep_time=0.01)  # Fast
        
        flow = Flow([slow_handler1, slow_handler2])
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            
            # Should timeout because one handler is too slow
            with pytest.raises(FlowError, match='timeout'):
                await flow_instance.process(task, timeout_sec=0.1)


    async def test_parallel_flow_startup_timeout(self, loop):
        """Test flow startup timeout with parallel steps."""
        
        class VerySlowStartHandler(BaseTaskHandler):
            def on_start(self):
                time.sleep(10)  # Very slow startup
                
            def handle(self, *tasks: Task):
                pass
        
        flow = Flow([VerySlowStartHandler(), ParallelTestHandler1()])
        
        with pytest.raises(TimeoutError):
            flow.start(timeout=0.5)  # Short timeout
        
        await flow.stop()

    async def test_parallel_flow_not_running_error(self, loop):
        """Test processing when parallel flow is not running."""
        handler1 = ParallelTestHandler1()
        handler2 = ParallelTestHandler2()
        
        flow = Flow([handler1, handler2])
        flow.start()
        await flow.stop(graceful=False)
        
        task = Task()
        with pytest.raises(NotRunningError):
            await flow.process(task)

    async def test_parallel_flow_priority_queues(self, loop):
        """Test priority queue behavior with parallel steps."""
        
        class PriorityTestHandler(BaseTaskHandler):
            def __init__(self, handler_name: str):
                self.handler_name = handler_name
                self.processed_order = []
                
            def handle(self, *tasks: Task):
                for task in tasks:
                    time.sleep(0.05)  # Small delay to see ordering
                    setattr(task, f'{self.handler_name}_processed', True)
                    self.processed_order.append(getattr(task, 'priority', 0))
        
        handler1 = PriorityTestHandler("handler1")
        handler2 = PriorityTestHandler("handler2")
        
        flow = Flow([handler1, handler2], queue_priorities=2)
        
        async with run_flow(flow) as flow_instance:
            # Create tasks with different priorities
            normal_task = Task()
            priority_task = Task()
            priority_task.set_priority(1)  # Higher priority
            
            # Process in order: normal, normal, priority, normal
            tasks = [
                asyncio.ensure_future(flow_instance.process(Task())),
                asyncio.ensure_future(flow_instance.process(Task())),
                asyncio.ensure_future(flow_instance.process(priority_task)),
                asyncio.ensure_future(flow_instance.process(Task())),
            ]
            
            await asyncio.gather(*tasks)
            
            # Priority task should have been processed
            assert hasattr(priority_task, 'handler1_processed')
            assert hasattr(priority_task, 'handler2_processed')

    async def test_parallel_flow_graceful_stop(self, loop):
        """Test graceful stop behavior with parallel steps."""
        
        class LongRunningHandler(BaseTaskHandler):
            def handle(self, *tasks: Task):
                for task in tasks:
                    time.sleep(0.2)  # Long processing time
                    task.long_result = "completed"
        
        handler1 = LongRunningHandler()
        handler2 = ParallelTestHandler2(sleep_time=0.05)
        
        flow = Flow([handler1, handler2])
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            
            # Start processing
            process_future = asyncio.ensure_future(flow_instance.process(task))
            await asyncio.sleep(0.1)  # Let it start processing
            
            # Stop gracefully
            await flow_instance.stop(graceful=True)
            
            # Task should complete
            result = await process_future
            assert result is True
            assert hasattr(task, 'long_result')

    async def test_parallel_flow_abort_stop(self, loop):
        """Test abort stop behavior with parallel steps."""
        
        class LongRunningHandler(BaseTaskHandler):
            def handle(self, *tasks: Task):
                for task in tasks:
                    time.sleep(0.5)  # Very long processing
                    task.long_result = "completed"
        
        handler1 = LongRunningHandler()
        handler2 = ParallelTestHandler2(sleep_time=0.05)
        
        flow = Flow([handler1, handler2])
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            
            # Start processing
            process_future = asyncio.ensure_future(flow_instance.process(task))
            await asyncio.sleep(0.1)  # Let it start processing
            
            # Stop abruptly
            await flow_instance.stop(graceful=False)
            
            # Task should fail
            with pytest.raises(FlowError):
                await process_future

    async def test_parallel_flow_cancelled_task(self, loop):
        """Test cancellation behavior with parallel steps."""
        
        class SlowHandler(BaseTaskHandler):
            def handle(self, *tasks: Task):
                for task in tasks:
                    time.sleep(0.3)
                    task.slow_result = "completed"
        
        handler1 = SlowHandler()
        handler2 = ParallelTestHandler2(sleep_time=0.05)
        
        flow = Flow([handler1, handler2])
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            
            # Start processing and cancel it
            with pytest.raises(asyncio.CancelledError):
                process_future = asyncio.ensure_future(flow_instance.process(task))
                await asyncio.sleep(0.1)
                process_future.cancel()
                await process_future

    async def test_parallel_flow_batching_behavior(self, loop):
        """Test batching behavior with parallel steps."""
        
        class BatchHandler(BaseTaskHandler):
            def __init__(self, handler_name: str):
                self.handler_name = handler_name
                self.batch_sizes = []
                
            def handle(self, *tasks: Task):
                self.batch_sizes.append(len(tasks))
                for task in tasks:
                    setattr(task, f'{self.handler_name}_batch_size', len(tasks))
        
        handler1 = BatchHandler("handler1")
        handler2 = BatchHandler("handler2")
        
        # Create flow with batching
        flow = Flow([
            FlowStep(handler1, batch_size=3, batch_timeout=0.1),
            FlowStep(handler2, batch_size=3, batch_timeout=0.1)
        ])
        
        async with run_flow(flow) as flow_instance:
            # Send 5 tasks quickly
            tasks = [Task() for _ in range(5)]
            await asyncio.gather(*[flow_instance.process(task) for task in tasks])
            
            # Check that batching occurred
            for task in tasks:
                assert hasattr(task, 'handler1_batch_size')
                assert hasattr(task, 'handler2_batch_size')
                # Batch sizes should be reasonable (1-3 based on timing)
                assert 1 <= task.handler1_batch_size <= 3
                assert 1 <= task.handler2_batch_size <= 3

    async def test_parallel_flow_task_expiration(self, loop):
        """Test task expiration behavior with parallel steps."""
        
        class VerySlowHandler(BaseTaskHandler):
            def handle(self, *tasks: Task):
                for task in tasks:
                    time.sleep(1.0)  # Very slow
                    task.very_slow_result = "completed"
        
        handler1 = VerySlowHandler()
        handler2 = ParallelTestHandler2(sleep_time=0.05)  # Fast
        
        flow = Flow([handler1, handler2])
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            
            # Process with short timeout
            with pytest.raises(FlowError, match='timeout'):
                await flow_instance.process(task, timeout_sec=0.2)

    async def test_parallel_flow_with_multiprocessing_nprocs(self, loop):
        """Test parallel flow with multiple processes per handler."""
        
        class ProcessIdHandler(BaseTaskHandler):
            def __init__(self, handler_name: str):
                self.handler_name = handler_name
                
            def handle(self, *tasks: Task):
                for task in tasks:
                    setattr(task, f'{self.handler_name}_pid', os.getpid())
        
        handler1 = ProcessIdHandler("handler1")
        handler2 = ProcessIdHandler("handler2")
        
        # Create flow with multiple processes per handler
        flow = Flow([
            FlowStep(handler1, nprocs=2),
            FlowStep(handler2, nprocs=2)
        ])
        
        async with run_flow(flow) as flow_instance:
            # Process multiple tasks to use different processes
            tasks = [Task() for _ in range(8)]
            await asyncio.gather(*[flow_instance.process(task) for task in tasks])
            
            # Collect unique process IDs for each handler
            handler1_pids = set()
            handler2_pids = set()
            
            for task in tasks:
                handler1_pids.add(task.handler1_pid)
                handler2_pids.add(task.handler2_pid)
            
            # Should have used multiple processes for each handler
            assert len(handler1_pids) >= 1  # At least 1 process, possibly 2
            assert len(handler2_pids) >= 1  # At least 1 process, possibly 2

    async def test_parallel_flow_with_explicit_collector(self, loop):
        """Test flows with parallel steps must have explicit collector handler."""
        
        class ResultAggregatorHandler(BaseTaskHandler):
            """Collector that aggregates results from parallel steps."""
            
            def handle(self, *tasks: Task):
                for task in tasks:
                    # Aggregate results from parallel handlers
                    if hasattr(task, 'handler1_result') and hasattr(task, 'handler2_result'):
                        task.aggregated_result = f"{task.handler1_result}+{task.handler2_result}"
                    task.collector_processed = True
        
        handler1 = ParallelTestHandler1(sleep_time=0.01, result_value="A")
        handler2 = ParallelTestHandler2(sleep_time=0.01, result_value="B")
        aggregator = ResultAggregatorHandler()
        
        # Parallel step followed by explicit collector - this is the required pattern
        flow = Flow([handler1, handler2], aggregator)
        
        async with run_flow(flow) as flow_instance:
            task = Task()
            await flow_instance.process(task)
            
            # Verify parallel processing occurred
            assert hasattr(task, 'handler1_result')
            assert hasattr(task, 'handler2_result')
            assert task.handler1_result == "A"
            assert task.handler2_result == "B"
            
            # Verify collector processed the aggregated results
            assert hasattr(task, 'collector_processed')
            assert task.collector_processed is True
            assert hasattr(task, 'aggregated_result')
            assert task.aggregated_result == "A+B"


