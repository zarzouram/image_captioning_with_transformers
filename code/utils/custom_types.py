from typing import TypeVar
import torch
import torch.utils.data

ModelType = TypeVar("ModelType", bound=torch.nn.Module)
OptimType = TypeVar("OptimType", bound=torch.optim.Optimizer)
SchedulerType = TypeVar("SchedulerType", bound=torch.optim.lr_scheduler.StepLR)
DeviceTye = TypeVar("DeviceTye", bound=torch.device)
DataIterType = TypeVar("DataIterType", bound=torch.utils.data.DataLoader)
