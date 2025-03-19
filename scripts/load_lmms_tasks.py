#!/usr/bin/env python3
"""
Pre-download script for lmms-eval datasets.

This script loads each evaluation dataset to ensure it's cached locally
before running on compute nodes without internet access.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Try to import lmms_eval

from lmms_eval.tasks import TaskManager, get_task_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-download lmms-eval datasets")
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="textvqa,mme,ai2d,ocrbench,m3exam,maxm,xgqa,cc-ocr-multi-lan",
        help="Comma-separated list of tasks to download"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    return parser.parse_args()

def preload_datasets(tasks):
    """Preload datasets for all specified tasks"""
    logger.info(f"Preloading datasets for tasks: {tasks}")
    
    # Initialize TaskManager
    task_manager = TaskManager(verbosity="INFO")
    
    # Load each task to trigger dataset caching
    for task in tasks:
        logger.info(f"Loading task: {task}")
        try:
            task_dict = get_task_dict(task, task_manager=task_manager)
            print(task_dict)
            task_obj = task_dict[task]
            task_obj.download()
            logger.info(f"Successfully loaded task: {task}")
        except Exception as e:
            logger.error(f"Failed to load task {task}: {e}")

def main():
    args = parse_args()
    
    # Parse task list
    task_list = [task.strip() for task in args.tasks.split(",")]
    logger.info(f"Tasks to download: {task_list}")
    
    preload_datasets(task_list)

if __name__ == "__main__":
    main()