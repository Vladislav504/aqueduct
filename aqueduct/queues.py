import multiprocessing as mp
from dataclasses import dataclass

from typing import List

from .handler import HandleConditionType
from .task import BaseTask


@dataclass
class FlowStepQueue:
    queue: mp.Queue
    handle_condition: HandleConditionType


def select_next_queue(queues: List[List[List[FlowStepQueue]]], task: BaseTask, start_index: int = 0) -> mp.Queue:
    for step_index in range(start_index, len(queues[task.priority])):
        step_queues = queues[task.priority][step_index]
        # For each step, try the first queue that matches the condition
        for step_queue in step_queues:
            if step_queue.handle_condition(task):
                return step_queue.queue
    # Fallback to last step's first queue
    return queues[task.priority][-1][0].queue


def find_target_queues(queues: List[List[List[FlowStepQueue]]], task: BaseTask, current_step: int) -> List[mp.Queue]:
    """
    Finds the target queues for a task based on handle conditions.
    
    Args:
        queues: 3-level queue structure [priority][step][queue_index]
        task: Task to find queues for
        current_step: Current step number (0-based, where we start searching from this step)
        
    Returns:
        List[mp.Queue]: List of target queues. If empty, task should go to final output.
    """
    priority_queues = queues[task.priority]
    
    # Try each step starting from current_step
    for step_index in range(current_step, len(priority_queues)):
        step_queues = priority_queues[step_index]
        
        if len(step_queues) > 1:
            # Parallel step - collect all matching queues
            target_queues = []
            for queue_info in step_queues:
                if queue_info.handle_condition(task):
                    target_queues.append(queue_info.queue)
            
            if target_queues:
                return target_queues
            # If no parallel queues accepted the task, continue to next step
            
        else:
            # Sequential step - try single queue
            queue_info = step_queues[0]
            if queue_info.handle_condition(task):
                return [queue_info.queue]
            # If condition fails, continue to next step
    
    # No step accepted the task - return final output queue
    final_queue = priority_queues[-1][0].queue
    return [final_queue]


def distribute_task_to_next_step(queues: List[List[List[FlowStepQueue]]], task: BaseTask, current_step: int) -> bool:
    """
    Distributes task to the next appropriate step queues (synchronous version for workers).
    
    Args:
        queues: 3-level queue structure [priority][step][queue_index]
        task: Task to distribute
        current_step: Current step number (0-based, where next step is current_step + 1)
    """
    target_queues = find_target_queues(queues, task, current_step + 1)
    
    for target_queue in target_queues:
        target_queue.put(task)
