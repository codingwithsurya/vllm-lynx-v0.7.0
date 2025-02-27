"""This file has the context manager for logging time metrics per batch using the cuda events library"""
import time
import torch
import wandb
import io
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from .cdf_sketch import CDFSketch
from ddsketch import DDSketch
from enum import Enum, auto
from typing import Union
from datetime import datetime
from vllm.config import VllmConfig
from transformers import MixtralConfig
from vllm.sequence import SequenceGroupMetadata

from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union


# define a constant for plotting dir
PLOTTING_DIR = "./profiling_output"


def if_enabled(func):
    def wrapper(self, *args, **kwargs):
        if not self.disable and self.profile_complete:
            return func(self, *args, **kwargs)
    return wrapper


class TaskType(Enum):
    TOKEN_REROUTING = auto()
    FUSED_MOE_KERNEL_CALL = auto()
    SCHEDULE_ITERATION = auto()
    WORKER_ITERATION = auto()
    E2E_REQUEST = auto()
    TOTAL_ATTN_COMPUTATION = auto()
    LAYERNORM_AFTER_ATTN = auto()
    TOTAL_EXPERT_COMPUTATION = auto()
    TOTAL_MLP_COMPUTATION = auto()
    # PREFILL_PRE_ATTN_PROJECTION = auto()
    # DECODE_PRE_ATTN_PROJECTION = auto()
    # PREFILL_POST_ATTN_PROJECTION = auto()
    # DECODE_POST_ATTN_PROJECTION = auto()
    # PREFILL_TOTAL_EXPERT_COMPUTATION = auto()
    # DECODE_TOTAL_EXPERT_COMPUTATION = auto()
    # PREFILL_TOTAL_ATTN_COMPUTATION = auto()
    # DECODE_TOTAL_ATTN_COMPUTATION = auto()
    # PREFILL_EXPERT_COMPUTATION = auto()
    # DECODE_EXPERT_COMPUTATION = auto()
    # PREFILL_ATTN_COMPUTATION = auto()
    # DECODE_ATTN_COMPUTATION = auto()
    BATCH_SIZE = auto()
    PREFILL_BATCH_SIZE = auto()
    DECODE_BATCH_SIZE = auto()
    EXPERTS_DROPPED = auto()

class TaskLoggingContextManagerGPU:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        MetricStore.get_instance().add_metric_gpu(self.task_type, self.start_event, self.end_event)
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        

class TaskLoggingContextManagerCPU:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        MetricStore.get_instance().add_metric_cpu(self.task_type, elapsed_time)
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        #print(f"{self.task_type} took {elapsed_time} s")

"""This class is a singleton. It stores the metrics for each task type."""
class MetricStore:
    _instance = None
    def __init__(self, vllm_config: VllmConfig) -> None:
        super(MetricStore, self).__init__()
        self.profile_complete = False
        self.batch_idx = 0
        self.metrics = {}
        self.raw_metrics = {}
        self.num_experts = -1
        self.top_k = -1
        self.hidden_size = -1
        self.num_prefill_batches = 0
        self.num_decode_batches = 0
        # self.disable = model_config.disable_latency_logging
        self.disable = False
        
        # Get the current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Define the full path including the base directory and the new subdirectory with date and time
        self.full_path = os.path.join(PLOTTING_DIR, f"plots_{current_datetime}")
        # Use os.makedirs to create the directory, including all necessary parent directories
        os.makedirs(self.full_path, exist_ok=True)
        
        

    @classmethod
    def get_instance(cls):
        assert cls._instance is not None, "Instance not created. Use get_or_create_instance() to create an instance."
        return cls._instance
    
    @classmethod
    def get_or_create_instance(cls, vllm_config: VllmConfig):
        if cls._instance is None:
            cls._instance = cls(vllm_config.model_config)
        return cls._instance
    
    def setting_params(self, config: MixtralConfig):
        self.num_experts=config.num_experts,
    
    
    def write_to_file(self, metrics: dict) -> None:
        with open("metrics.txt", "a") as f:
            f.write(str(metrics))
            f.write("\n")
        # print the metrics dictionary
        print(metrics)
        # self.metrics = {}
        
    #### Important functions for config manager ####
    def mark_profiling_done(self):
        self.profile_complete = True
        # print("The profiling is complete and the flag is being set for metric logging")


    def on_batch_end_worker(self):
        self.batch_idx += 1
        self.calculate_elapsed_times()
        # if self.batch_idx % 4 == 0 and self.profile_complete:
        
    def on_batch_start_worker(self, seq_group_metadata_list: Optional[List[SequenceGroupMetadata]]):
        if not seq_group_metadata_list:
            return
        self.put_metric(TaskType.BATCH_SIZE, len(seq_group_metadata_list))
        is_prefill = all([seq_group_metadata.is_prompt for seq_group_metadata in seq_group_metadata_list])
        if is_prefill:
            self.num_prefill_batches += 1
            self.put_metric(TaskType.PREFILL_BATCH_SIZE, len(seq_group_metadata_list))
            # print(f"Prefill batch size: {len(seq_group_metadata_list)}")
        else:
            self.num_decode_batches += 1
            self.put_metric(TaskType.DECODE_BATCH_SIZE, len(seq_group_metadata_list))
            # print(f"Decode batch size: {len(seq_group_metadata_list)}")

    ##################################################


            
    @if_enabled
    def plot_batch_type(self):
        # create a matplotlib plot with one bar for num_prefill_batches, one bar for num_decode_batches and one for their sum
        plt.bar(["Prefill", "Decode", "Total"], [self.num_prefill_batches, self.num_decode_batches, self.num_prefill_batches + self.num_decode_batches])
        plt.title("Number of batches per type")
        wandb.log({"Number of batches per type": plt})
        
        
    @if_enabled
    def add_metric_gpu(self, task_type: TaskType, start_event: torch.cuda.Event, end_event: torch.cuda.Event) -> None:
        if task_type in self.metrics:
            self.raw_metrics[task_type].append((start_event, end_event))
        else:
            self.raw_metrics[task_type] = [(start_event, end_event)]
        # print(f"Added metric for {task_type} where the start event is {start_event} and the end event is {end_event}")
       
    @if_enabled
    def put_metric(self, task_type: TaskType, metric_value: float) -> None:
        if task_type not in self.metrics:
            self.metrics[task_type] = CDFSketch(metric_name=task_type, save_table_to_wandb=False)
        self.metrics[task_type].put(metric_value)
    
    @if_enabled
    def add_metric_cpu(self, task_type: TaskType, elapsed_time: float) -> None:
        self.put_metric(task_type, elapsed_time)
    
    @if_enabled
    def get_metrics(self) -> dict:
        return self.metrics
    
    @if_enabled
    def calculate_elapsed_times(self):
        torch.cuda.synchronize()  # Ensure all prior operations on the default stream are complete before calculations
        # print("Calculating elapsed times")
        for task_type, measurements in list(self.raw_metrics.items()):
            if task_type == TaskType.SCHEDULE_ITERATION:
                continue  # Skip processing for specific task types if needed
            
            for measurement in measurements:
                start_event, end_event = measurement  # Unpack the tuple
                try:
                    # Calculate elapsed time. Ensure both are CUDA events; otherwise, an exception will be thrown
                    elapsed_time = start_event.elapsed_time(end_event)
                    self.add_metric_cpu(task_type, elapsed_time)
                except AttributeError as e:
                    print(f"Error processing measurement for {task_type}: {e}")
                    # Handle error, e.g., by skipping this measurement or logging the issue
            
            # Update the metrics dictionary with processed measurements (either original values or calculated elapsed times)
            self.raw_metrics[task_type] = []


    def set_disable(self):
        self.disable = True


    @if_enabled
    def plot_metrics(self):
        self.plot_batch_type()
        self.write_to_file(self.metrics)
        for task_type, measurements in self.metrics.items():
            if task_type == TaskType.BATCH_SIZE:
                measurements.plot_cdf(self.full_path, f"{task_type}","size")
            else:
                measurements.plot_cdf(self.full_path, f"{task_type}_execution_time","time")
