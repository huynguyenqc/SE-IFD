import os
import subprocess
import sys
import time
import torch
from typing import List


def main(args_list: List[str]):
    num_gpus = torch.cuda.device_count()
    # args_list.append("--group_name=group_{}".format(time.strftime("%Y_%m_%d-%H%M%S")))
    group_name = f'group_{time.strftime("%Y_%m_%d-%H%M%S")}'
    args_list += ['--group-name', group_name]
    stdout_dir = os.path.join('logs/distributed/', group_name)
    if not os.path.isdir(stdout_dir):
        os.makedirs(stdout_dir)
        os.chmod(stdout_dir, 0o775)

    workers = []
    for i in range(num_gpus):
        args_list_i = args_list + ['--rank', f'{i}']
        stdout = None if i == 0 else open(
            os.path.join(stdout_dir, "GPU_{}.log".format(i)), "w")
        print(args_list)
        p = subprocess.Popen(args_list_i, stdout=stdout)
        workers.append(p)

    for p in workers:
        p.wait()


if __name__ == '__main__':
    main(sys.argv[1: ])