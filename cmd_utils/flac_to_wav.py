import argparse
import os
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--ext', type=str, required=True)

args = parser.parse_args()

file_list = sorted(sum([[
    str(f)
    for f in Path(
        pth).rglob('*.{}'.format(args.ext))] for pth in args.dir.split(',')], []))

for fp in file_list:
    subprocess.call([
        'ffmpeg',
        '-i',
        fp,
        '-c:a',
        'pcm_s24le',
        fp[: -len(args.ext)] + 'wav'
    ])
    os.remove(fp)