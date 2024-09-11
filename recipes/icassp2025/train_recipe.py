import argparse
import gc
import numpy as np
import os
import random
import torch
import yaml
from pydantic import BaseModel
from typing import List, Optional

from .trainer.main import train_function, TrainConfigs, ModelConfigs


torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)


class StageConfigs(BaseModel):
    identifier: str
    train_configs: str
    net_configs: str


class RecipeConfigs(BaseModel):
    stages: List[StageConfigs]


def single_stage_training(
        log_dir: str,
        identifier: str,
        train_configs_path: str,
        model_configs_path: str,
        pretrain_model_configs_path: Optional[str] = None,
        pretrain_model_path: Optional[str] = None,
        rank: int = 0,
        num_gpus: int = 1,
        group_name: str = '') -> str:
    
    with open(train_configs_path, 'r') as f:
        train_configs = yaml.safe_load(f)
    with open(model_configs_path, 'r') as f:
        model_configs = yaml.safe_load(f)
    train_configs = TrainConfigs(**train_configs)
    model_configs = ModelConfigs(**model_configs)

    if pretrain_model_configs_path is not None:
        with open(pretrain_model_configs_path, 'r') as f:
            pretrain_model_configs = yaml.safe_load(f)
        pretrain_model_configs = ModelConfigs(**pretrain_model_configs)
    else:
        pretrain_model_configs = None
        # pretrain_model_configs = ModelConfigs(**pretrain_model_configs)

    return train_function(log_parent_dir=log_dir,
                          identifier=identifier,
                          train_configs=train_configs,
                          model_configs=model_configs,
                          pretrain_model_configs=pretrain_model_configs,
                          pretrain_model_path=pretrain_model_path,
                          num_gpus_=num_gpus,
                          rank_=rank,
                          group_name_=group_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--rank', type=int, required=False, default=0)
    parser.add_argument('--group-name', type=str, required=False, default='')

    args = parser.parse_args()
    with open(args.configs, 'r') as f:
        configs = yaml.safe_load(f)

    configs = RecipeConfigs(**configs)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print('Warning: Training on 1 GPU!')
            num_gpus = 1
        else:
            print(f'Run distributed training on {num_gpus} GPUs')

    pretrain_model_configs_path = None
    pretrain_model_path = None

    for stage_cfgs in configs.stages:
        training_log_dir = single_stage_training(
            log_dir=args.log_dir,
            identifier=stage_cfgs.identifier,
            train_configs_path=stage_cfgs.train_configs,
            model_configs_path=stage_cfgs.net_configs,
            pretrain_model_configs_path=pretrain_model_configs_path,
            pretrain_model_path=pretrain_model_path,
            num_gpus=num_gpus,
            rank=args.rank,
            group_name=args.group_name)

        gc.collect()                # garbage collector
        torch.cuda.empty_cache()    # release allocated model

        # Configs of previous stage is pretrain configs for next stage
        pretrain_model_configs_path = stage_cfgs.net_configs
        # Use the weight of the last epoch as pretraining weight for next stage
        pretrain_model_path = os.path.join(training_log_dir, 'checkpoints', 'epoch_final.pt')


if __name__ == '__main__':
    main()
