
from typing import List, Optional, Dict

from .logger import log
from .queues import FlowStepQueue
from .task import BaseTask



class TaskCollectionInfo:
    """Information for tracking parallel task collection."""
    
    def __init__(self, expected_count: int):
        self.recieved_results: List[BaseTask] = []
        self.expected_count: int = expected_count
    
    def add_result(self, task: BaseTask):
        """Add a result to the collection."""
        self.recieved_results.append(task)
    
    def is_complete(self) -> bool:
        """Check if we have all expected results."""
        return len(self.recieved_results) >= self.expected_count


class ParallelTaskCollector:
    """Handles collection and assembly of parallel task results."""
    
    def __init__(self, step_number: int, queues: List[List[List[FlowStepQueue]]]):
        self._task_collections: Dict[str, TaskCollectionInfo] = {}
        self._collect_parallel_results = False
        self._parallel_step_conditions = []


        # Configure parallel task collector if this worker needs to collect results from previous parallel step
        # Only needed for step 2 and beyond (step 1 has no previous step to collect from)
        if step_number >= 2:
            # Calculate previous step's queue index
            # Step numbering is 1-indexed, but queue arrays are 0-indexed
            # If we're step N, we read from queue index N-1, and previous step uses index N-2
            prev_step_index = step_number - 2
            if prev_step_index < len(queues[0]):
                # Get the queue configuration for the previous step
                # This tells us about parallel workers that feed into this step
                prev_step_queues = queues[0][prev_step_index]
                # Configure collector with previous step's parallel conditions
                # This enables counting expected results and assembling parallel outputs
                if len(prev_step_queues) > 1:
                    self._collect_parallel_results = True
                    self._parallel_step_conditions = [q.handle_condition for q in prev_step_queues] 
    
    def get_expired_result(self, task: BaseTask) -> Optional[BaseTask]:
        """Check for expired parallel tasks and assemble them."""
        if not self._collect_parallel_results:
            return None
            
        collection_info: Optional[TaskCollectionInfo] = self._task_collections.pop(key=task.task_id, default=None)
        if not collection_info:
            return None

        # Assemble whatever partial results we have and store them
        if collection_info.recieved_results:
            assembled_task = self._assemble_results(collection_info.recieved_results)
            if assembled_task:
                return assembled_task
        
        return None

    
    def get_result(self, task: BaseTask) -> Optional[BaseTask]:
        """Collect parallel results and complete when all expected results are received."""
        if not self._collect_parallel_results:
            return task
        
        task_id = task.task_id
        
        if task_id not in self._task_collections:
            expected_count = self._count_expected_results(task)
            self._task_collections[task_id] = TaskCollectionInfo(expected_count)
        
        collection_info = self._task_collections[task_id]
        collection_info.add_result(task)
        
        if collection_info.is_complete():
            final_task = self._assemble_results(collection_info.recieved_results)
            del self._task_collections[task_id]
            
            return final_task
        
        return None
    
    def _count_expected_results(self, task: BaseTask) -> int:
        """Count how many parallel steps should process this specific task."""
        if not self._collect_parallel_results:
            return 1
        
        count = 0
        for condition in self._parallel_step_conditions:
            if condition(task):
                count += 1
        
        return count
    
    def _assemble_results(self, results: List[BaseTask]) -> Optional[BaseTask]:
        """Assemble results from parallel steps"""
        if not results:
            return None
        
        # last task immutable data wins
        final_task = results[-1]
        
        for result in results[:-1]:
            final_task.update(result)
            if final_task.metrics and result.metrics:
                final_task.metrics.extend(result.metrics)

        return final_task
