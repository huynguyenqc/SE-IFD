import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--dir-1', type=str, required=True)
parser.add_argument('--dir-2', type=str, required=True)
parser.add_argument('--ext', type=str, required=True)
parser.add_argument('--out', type=str, required=True)

args = parser.parse_args()

file_list_1 = sorted([
    str(f)
    for f in Path(
        args.dir_1).rglob('*.{}'.format(args.ext))])

file_list_2 = sorted([
    str(f)
    for f in Path(
        args.dir_2).rglob('*.{}'.format(args.ext))])

file_list = [
    item 
    for pair_item in zip(file_list_1, file_list_2) 
    for item in pair_item]

with open(args.out, 'w') as f_out:
    f_out.write('\n'.join(file_list) + '\n')
