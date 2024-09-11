import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    args = parser.parse_args()

    x = torch.tensor(args.x).cuda()
    y = torch.tensor(args.y).cuda()

    print(x)
    print(y)
    print(x + y)
