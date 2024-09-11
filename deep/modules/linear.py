import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any

from deep.base_module import ModuleInterface
from deep.lora import LoRAInterface, LoRAConfigurations


class _Linear(nn.Linear):
    def __init__(
            self, in_features: int, out_features: int, 
            bias: bool = True, device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)


class Linear(_Linear, ModuleInterface, LoRAInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        in_features: int
        out_features: int
        bias: bool = True
        device: Any = None
        dtype: Any = None
        LoRA: LoRAConfigurations = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        _Linear.__init__(self, *args, **kwargs)
        LoRAInterface.__init__(self)

        if self._configs.LoRA is not None:
            in_dim = self.in_features
            out_dim = self.out_features
            rank_dim = math.floor(self._configs.LoRA.rank_ratio * self.max_rank_dim(in_dim, out_dim))

            # LoRA parameters
            if rank_dim > 0:
                self.lora_A, self.lora_B = self.new_LoRA_paramters(self.weight, in_dim, out_dim, rank_dim)
                self.scaling = self._configs.LoRA.alpha / rank_dim
                self.reset_LoRA_parameters(self.lora_A, self.lora_B)
            else:
                self.lora_A, self.lora_B, self.scaling = None, None, None

            self.weight.requires_grad_(False)
    
    def forward(self, x__d: torch.Tensor) -> torch.Tensor:
        if  self._configs.LoRA is None or self.lora_A is None:
            adapted_weight = self.weight
        else:
            delta_weight = self.lora_B @ self.lora_A
            adapted_weight = self.weight + self.scaling * delta_weight
        return F.linear(x__d, adapted_weight, self.bias)

    def LoRA_merge_(self) -> None:
        if  self._configs.LoRA is not None and self.lora_A is not None:
            # Update weight by adding adaptation amount
            with torch.no_grad():
                delta_weight = self.lora_B @ self.lora_A
                self.weight += self.scaling * delta_weight

            # Reset adaptation amount
            self.reset_LoRA_parameters(self.lora_A, self.lora_B)