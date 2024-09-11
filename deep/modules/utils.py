import torch
from typing import Tuple


def inverse_square_root_2x2(
        A__: torch.Tensor,
        B__: torch.Tensor,
        D__: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Compute U^{-1/2}

        U =  [ A__  B__ ]
             [ B__  D__ ]

    """
    # Matrix Inverse Square Root U = V^-0.5
    # sqrt of a 2x2 matrix,
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    tau__ = A__ + D__
    delta__ = torch.addcmul(
        input=A__*D__, tensor1=B__, tensor2=B__, value=-1)
    s__ = delta__.sqrt()
    t__ = (tau__ + 2*s__).sqrt()

    # matrix inverse, 
    # http://mathworld.wolfram.com/MatrixInverse.html
    rst__ = (s__ * t__).reciprocal()
    iA__ = (s__ + D__) * rst__
    iD__ = (s__ + A__) * rst__
    iB__ = - B__ * rst__
    return (iA__, iB__, iD__)


def sanity_check():
    a = torch.tensor([1], dtype=torch.float32) 
    b = torch.tensor([-0.5], dtype=torch.float32)
    d = torch.tensor([4], dtype=torch.float32)

    ia, ib, id = inverse_square_root_2x2(a, b, d)

    mat_1 = torch.tensor(
        [[a[0].item(), b[0].item()],
         [b[0].item(), d[0].item()]])
    mat_2 = torch.tensor(
        [[ia[0].item(), ib[0].item()],
         [ib[0].item(), id[0].item()]])

    print(mat_1)
    print(mat_2)
    print(mat_1 @ (mat_2 @ mat_2))
    print((mat_2 @ mat_2) @ mat_1)


if __name__ == '__main__':
    sanity_check()
