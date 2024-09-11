import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--in-files', type=str, required=True, nargs='+')
parser.add_argument('--out-file', type=str, required=True)

args = parser.parse_args()

with open(args.out_file, 'w') as f_out:
    for in_file in args.in_files:
        with open(in_file, 'r') as f_in:
            f_out.write(f_in.read())
