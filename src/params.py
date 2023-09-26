from typing import Optional
from typing import Tuple, List

import attr


@attr.s(auto_attribs=True)
class MAEParams:
    lr: float = 0.0004
    batch_size: int = 32
    val_batch_size: int = 32    
    resolution: Tuple[int, int] = (224, 224)
    data_root: str = "/home/dl/takamagahara/hutodama/MAE/data/mvtec_loco"
    category: str = "breakfast_box"
    gpus: int = 2
    max_epochs: int = 100
    num_sanity_val_steps: int = 10
    check_val_every_n_epoch : int = 10
    ckpt_every_n_epoch : int = 50
    ckpt_period : int = 2
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    is_logger_enabled: bool = False
    is_verbose: bool = True
    num_workers: int = 4
    num_samples: int = 8
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
    arch: str = "mae_vit_base_patch16"
    ckpt_dir: str = "/home/dl/takamagahara/hutodama/MAE/ckpt/mae_pretrain_vit_base_full.pth"
    
    pretrained_model_ckpt: str = "/home/dl/takamagahara/hutodama/MAE/src/wandb/run-20230807_101801-kuia1oij/ckpt/last.ckpt"
    eval_iter: int = 5