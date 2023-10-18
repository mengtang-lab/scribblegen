from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING

class DatasetEnum(Enum):
    ADE20K = 0
    PascalSegmentation = 1
    PascalScribble = 2

@dataclass
class ExpConfig:
    max_epochs: int = MISSING
    batch_size: int = MISSING
    learning_rate: float = MISSING
    dataset: DatasetEnum = MISSING
    split_path: Optional[str] = None

    gpus: List[int] = MISSING
    log_dir: str = MISSING

    accumulate_grad_batches: int = 1
    image_size: Tuple[int, int] = (512, 512)
    one_hot_labels: bool = False
    drop_out_rate: float = 0
    drop_out_text: Optional[str] = None  # the prompt to use when dropped out, or None to not drop out prompt at all
    class_hint: bool = False  # whether to include class names in the prompt, currently only for pascal scribbles
    overfit: bool = False
    num_workers: int = 4
    
    sd_locked: bool = True  # whether to train bottom half of SD model
    only_mid_control: bool = False
    
    model_path: str = './ControlNet/models/control_sd15_ini.ckpt'
    model_config_path: str = './ControlNet/models/cldm_v15.yaml'
    resume_path: Optional[str] = None
    resume_init_only: bool = False  # whether to use `resume_path` only as a model initializaiton (i.e., don't resume training)
    logger_freq: int = 300
