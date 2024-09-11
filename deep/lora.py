import math
import torch
from torch import nn
from pydantic import BaseModel
from typing import Optional, Tuple


class _LoRAConfigurations(BaseModel):
    r"""
    Configurations for Low-Rank Adaptation Method (LoRA)

    LoRA is a technique that allows fine-tuning a linear transformation
    with weight matrix :math:`\mathbf{W} \in \mathbb{R}^{m \times n}`,
    where :math:`n` and :math:`m` are dimensions of input and output,
    respectively. LoRA aims to learn the amount of modification of the
    weight, i.e., :math:`\Delta\mathbf{W}`, which can be decomposed
    using low-rank decomposition; in other words,

    .. math::
        \Delta\mathbf{W} = \mathbf{B}\mathbf{A}

    where :math:`\mathbf{A} \in \mathbb{R}^{r \times n}`,
    :math:`\mathbf{B} \in \mathbb{R}^{m \times r}`, and :math:`r` is
    the rank of :math:`\Delta\mathbf{W}`.
    The adapted mat

    To keep the learned dimension lower than the original dimension,
    we need
    
    .. math::
        mr + rn < mn \Leftrightarrow r < \frac{mn}{m+n}

    In this implementation, instead of defining the rank :math:`r`
    directly, we define the rank ratio :math:`\rho < 1`, so that
    .. math::
        r = \left\lfloor \rho\frac{mn}{m + n} \rfloor\right

    Args:
        rank_ratio (float): The rank ratio of the maximum rank.
            Default to 1.
        alpha (float): Determine LoRA scale.
            Default to 1.
    """
    rank_ratio: float = 1.0
    alpha: float = 1.0


LoRAConfigurations = Optional[_LoRAConfigurations]


class LoRAInterface:
    def __init__(self, *args, **kwargs) -> None:
        assert isinstance(self, nn.Module), \
            'Only apply LoRA interface to `nn.Module`!'

    @staticmethod
    def max_rank_dim(in_dim: int, out_dim: int) -> int:
        return (in_dim * out_dim) // (in_dim + out_dim)

    @staticmethod
    def reset_LoRA_parameters(
        LoRA_A_rn: torch.Tensor,
        LoRA_B_mr: torch.Tensor) -> None:
        """
        Args:
            LoRA_A_rn (torch.Tensor): :math:`\mathbf{A}` matrix of LoRA
            LoRA_B_mr (torch.Tensor): :math:`\mathbf{B}` matrix of LoRA
        """
        nn.init.kaiming_uniform_(LoRA_A_rn, a=math.sqrt(5))
        nn.init.zeros_(LoRA_B_mr)

    @staticmethod
    def new_LoRA_paramters(
        weight: torch.Tensor, 
        in_dim: int, 
        out_dim: int, 
        rank_dim: int) -> Tuple[nn.Parameter, nn.Parameter]:
        """
        Args:
            weight (torch.Tensor): The weight tensor :math:`\mathbf{W}`.
                The tensor configurations, e.g., `dtype` and `device`
                will be used to initialise the parameters.
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            rank_dim (int): Rank dimension
        Returns:
            Tuple[nn.Parameter, nn.Parameter]: The matrices 
                :math:`\mathbf{A}` and :math:`\mathbf{B}`, respectively.
        """
        lora_A_rn = nn.Parameter(weight.new_zeros((rank_dim, in_dim)))
        lora_B_mr = nn.Parameter(weight.new_zeros((out_dim, rank_dim)))
        return lora_A_rn, lora_B_mr

    def LoRA_merge_(self) -> None:
        raise NotImplementedError(
            'This function must be implemented in each subclass!')


def LoRA_merge_module_(module: nn.Module) -> nn.Module:
    assert isinstance(module, nn.Module), \
        'The input `module` must be a `nn.Module`!'
    for sub_module in module.modules():
        if isinstance(sub_module, LoRAInterface):
            sub_module.LoRA_merge_()

    return module