import argparse
from argparse import ArgumentParser

from params import MAEParams
from train import train

def parse_args() -> ArgumentParser:
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_root", type=str, default="")
    arg_parser.add_argument("--category", type=str, default="")
    arg_parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    arg_parser.add_argument("--arch", type=str, default="mae_vit_base_patch16")
    arg_parser.add_argument("--pretrained_model_ckpt", type=str, default="")
    arg_parser.add_argument("--resolution", type=int, nargs=2, default=[224, 224])
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--num_workers", type=int, default=4)
    arg_parser.add_argument("--lr", type=float, default=0.0004)
    arg_parser.add_argument("--max_epochs", type=int, default=100)
    arg_parser.add_argument("--num_sanity_val_steps", type=int, default=10)
    arg_parser.add_argument("--ckpt_every_n_epoch", type=int, default=50)
    arg_parser.add_argument("--ckpt_period", type=int, default=2)
    arg_parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    arg_parser.add_argument("--weight_decay", type=float, default=0.0)
    arg_parser.add_argument("--is_logger_enabled", type=bool, default=False)
    arg_parser.add_argument("--is_verbose", type=bool, default=True)
    arg_parser.add_argument("--warmup_steps_pct", type=float, default=0.02)
    arg_parser.add_argument("--decay_steps_pct", type=float, default=0.2)
    arg_parser.add_argument("--gpus", type=int, default=2)
    arg_parser.add_argument("--num_samples", type=int, default=8)
    arg_parser.add_argument("--val_batch_size", type=int, default=32)
    arg_parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    
    return arg_parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    train(args)