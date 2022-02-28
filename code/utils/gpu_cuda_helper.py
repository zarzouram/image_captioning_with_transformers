# Reference
import subprocess
from typing import List, Tuple
import torch


def get_gpu_memory() -> List[int]:
    """Get the current gpu usage.
    Reference: https://stackoverflow.com/a/49596019

    Returns
    -------
    usage: list
        Values of memory usage per GPU as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ])
    # Convert lines into a dictionary
    result = result.decode('utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]

    return gpu_memory


def get_gpus_avail() -> List[Tuple[int, float]]:
    """Get the GPU ids that have memory usage less than or equal to 40%
    """
    memory_usage = get_gpu_memory()

    memory_usage_percnt = [m / 11178 for m in memory_usage]
    cuda_ids = [(i, m) for i, m in enumerate(memory_usage_percnt) if m <= 0.4]

    header = ["cuda id", "Memory usage"]
    no_gpu_mssg = "No available GPU"
    if cuda_ids:
        print(f"{header[0]:^10}{header[1]:^15}")
        for (idx, m) in cuda_ids:
            print(f"{idx:^10}{m:^15.2%}")
    else:
        print(f"{no_gpu_mssg:-^25}")
        print(f"{header[0]:^10}{header[1]:^15}")
        for idx, m in enumerate(memory_usage_percnt):
            print(f"{idx:^10}{m:^15.2%}")

    return sorted(cuda_ids, key=lambda tup:
                  (tup[1], -tup[0])) if cuda_ids else cuda_ids


def select_device(device: str = "gpu"):
    if device == "cpu":
        return torch.device(device)
    elif device == "gpu":
        gpus_avail = get_gpus_avail()
        if gpus_avail:
            return torch.device(f"cuda:{gpus_avail[0][0]}")
        else:
            return torch.device("cpu")


if __name__ == "__main__":
    print(select_device("cpu"))
    print(select_device("gpu"))
