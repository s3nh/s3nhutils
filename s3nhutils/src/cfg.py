import os

from typing import List, Dict, Union
from typing import TypeVar, Any, Tuple


class InferenceConfig:
    image_path: str = 'data'
    size: int = 512
    model_path: str = 'model_path'

class MainConfig: 
    apex: bool = False
    debug: bool = False
    print_freq: int = 10
    #Number of workers
    num_workers: int = 4
    # This is main feature
    model_name: str = 'convmixer_768_32'
    size: Union[int, Tuple[int, int]] = 512
    datapath: str = 'assets/data' 
    extension: Union[str, Tuple[str]]
    scheduler: str = 'CosingAnnealingLR' 
    n_epochs: int = 35
    T_max: int = 6 #CosineAnnealingLR

    lr: float = 1e-4
    min_lr: float = 1e-6
    batch_size: int = 4
    weight_decay: float = 1e-6
    gradient_accumulation_steps: int = 1
    max_grad_norm: int = 1000
    seed: int = 42
    target_size: int = 1
    target_col: str = 'target'
    n_fold: int = 4
    trn_fold: List = [0, 1, 2, 3]
    train: bool = True
    output_dir: str = f"{model_name}_{n_epochs}_{batch_size}_{n_fold}_{scheduler}"
    os.makedirs(output_dir, exist_ok = True)
