import argparse
import numpy as np
import random
import torch
import yaml

from .trainer.main import train_function, TrainConfigs, ModelConfigs


random.seed(3407)
torch.manual_seed(3407)
np.random.seed(3407)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--identifier', type=str, required=True)
    parser.add_argument('--train-configs', type=str, required=True)
    parser.add_argument('--model-configs', type=str, required=True)
    parser.add_argument('--init-state-dict-path', type=str, required=False, default=None)
    parser.add_argument('--pretrain-model-configs', type=str, required=False, default=None)
    parser.add_argument('--pretrain-model-path', type=str, required=False, default=None)
    parser.add_argument('--rank', type=int, required=False, default=0)
    parser.add_argument('--group-name', type=str, required=False, default='')

    args = parser.parse_args()
    with open(args.train_configs, 'r') as f:
        train_configs = yaml.safe_load(f)
    with open(args.model_configs, 'r') as f:
        model_configs = yaml.safe_load(f)
    train_configs = TrainConfigs(**train_configs)
    model_configs = ModelConfigs(**model_configs)
    identifier = args.identifier

    print(model_configs.configs.model_dump().keys())

    if args.pretrain_model_configs is not None:
        with open(args.pretrain_model_configs, 'r') as f:
            pretrain_model_configs = yaml.safe_load(f)
        pretrain_model_configs = ModelConfigs(**pretrain_model_configs)
    else:
        pretrain_model_configs = None
        # pretrain_model_configs = ModelConfigs(**pretrain_model_configs)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print('Warning: Training on 1 GPU!')
            num_gpus = 1
        else:
            print(f'Run distributed training on {num_gpus} GPUs')

    train_function(log_parent_dir=args.log_dir,
                   identifier=identifier,
                   train_configs=train_configs,
                   model_configs=model_configs,
                   init_state_dict_path=args.init_state_dict_path,
                   pretrain_model_configs=pretrain_model_configs,
                   pretrain_model_path=args.pretrain_model_path,
                   num_gpus_=num_gpus,
                   rank_=args.rank,
                   group_name_=args.group_name)


if __name__ == '__main__':
    main()
