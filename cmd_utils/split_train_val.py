import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--file-list', type=str, required=True)
parser.add_argument('--train-name', type=str, required=True)
parser.add_argument('--val-name', type=str, required=True)
parser.add_argument('--out-train-list', type=str, required=True)
parser.add_argument('--out-val-list', type=str, required=True)

args = parser.parse_args()

with open(args.file_list, 'r') as f:
    lines = [l.strip() for l in f.readlines()]
    file_list = [lines[i: i+2] for i in range(0, len(lines), 2)]

with open(args.train_name, 'r') as f:
    train_subset = [l.strip() for l in f.readlines()]

with open(args.val_name, 'r') as f:
    val_subset = [l.strip() for l in f.readlines()]

train_list = [
    sitem
    for item in file_list
    if os.path.basename(item[0]).split('.', 1)[0] in train_subset
    for sitem in item]

val_list = [
    sitem
    for item in file_list
    if os.path.basename(item[0]).split('.', 1)[0] in val_subset
    for sitem in item]


with open(args.out_train_list, 'w') as f_out:
    f_out.write('\n'.join(train_list) + '\n')


with open(args.out_val_list, 'w') as f_out:
    f_out.write('\n'.join(val_list) + '\n')
