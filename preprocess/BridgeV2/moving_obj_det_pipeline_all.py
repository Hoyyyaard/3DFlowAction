import os
import subprocess
from tqdm import tqdm
import hydra
import omegaconf
import zarr
import sys
from glob import glob

def assign_task_bounds_to_gpus(n_tasks, n_gpus):
    """
    Assigns task ID bounds to GPUs as evenly as possible.

    Parameters:
    n_tasks (int): Number of tasks to be distributed.
    n_gpus (int): Number of GPUs available.

    Returns:
    list: A list of tuples where each tuple represents the lower (inclusive)
          and upper (exclusive) bounds of task IDs for that GPU.
    """
    # Calculate the base number of tasks per GPU and the remainder
    tasks_per_gpu = n_tasks // n_gpus
    remainder = n_tasks % n_gpus

    # Initialize the starting task ID
    start_id = 0

    # Distribute tasks to GPUs
    task_bounds = []
    for i in range(n_gpus):
        # Determine the number of tasks for this GPU
        num_tasks = tasks_per_gpu + (1 if i < remainder else 0)
        # Assign the bounds
        task_bounds.append((start_id, start_id + num_tasks))
        # Update the starting ID for the next GPU
        start_id += num_tasks

    return task_bounds

def main():
    avaliable_gpu = [0, 1, 2, 3, 4, 5, 6, 7]
    num_gpu = 8
    buffer = glob("< Your path to processed BridgeV2 >/*")
    buffer = sorted(buffer, key=lambda x: int(x.split("/")[-1]))
    num_episode = len(buffer)
    print(f"Total number of episodes: {num_episode}")
    task_bounds = assign_task_bounds_to_gpus(num_episode, num_gpu)
    processes = []
    for i, (start, end) in enumerate(task_bounds):
        # Start a new process for each task range
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(avaliable_gpu[i])
        process = subprocess.Popen(
            [
                "python",
                "preprocess/BridgeV2/moving_obj_det_pipeline.py",
                f"--start_idx={start}",
                f"--end_idx={end}",
                "--data_path=< Your path to processed BridgeV2 >",
                "--data_buffer_path=< Zarr output path >"
            ],
            env=env,
        )
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
