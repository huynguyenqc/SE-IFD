import argparse
import soundfile as sf
import warnings
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--ext', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--sr', type=int, required=True)
parser.add_argument('--min-length', type=float, required=False, default=-1.0)

args = parser.parse_args()

min_length = int(args.min_length * args.sr)
kept_file_list = []

for pth in args.dir.split(','):
    for f in Path(pth).rglob('*.{}'.format(args.ext)):
        file_path = str(f)
        try:
            with sf.SoundFile(file_path) as sf_obj:
                assert sf_obj.samplerate == args.sr, 'Sampling rate must match!'
                if args.min_length > 0:
                    assert sf_obj.frames > min_length, \
                        'Number of frames must be more than {} seconds!'.format(
                            args.min_length)
                kept_file_list.append(file_path)
        except Exception as e:
            warnings.warn('## Cannot read `{}`\n  Error: {}'.format(
                file_path, e))

# file_list = sorted(sum([[
#     str(f)
#     for f in Path(
#         pth).rglob('*.{}'.format(args.ext))] for pth in args.dir.split(',')], []))
# 
# if args.sr != 0:
#     kept_file_list = []
#     for fp in file_list:
#         with sf.SoundFile(fp) as sf_obj:
#             if sf_obj.samplerate == args.sr:
#                 kept_file_list.append(fp)
# 

with open(args.out, 'w') as f_out:
    f_out.write('\n'.join(sorted(kept_file_list)) + '\n')
