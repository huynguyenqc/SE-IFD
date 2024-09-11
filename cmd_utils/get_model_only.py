import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--in-checkpoint', type=str, required=True)
parser.add_argument('--out-checkpoint', type=str, required=True)

args = parser.parse_args()
data = torch.load(args.in_checkpoint, map_location='cpu')
model_only = {'model': data['model']}
torch.save(model_only, args.out_checkpoint)