import torch
from torch import nn
from typing import Any

from deep.base_module import ModuleInterface


class LSTMConstructorArgs(ModuleInterface.ConstructorArgs):
    input_size: int
    hidden_size: int
    num_layers: int = 1
    bias: bool = True
    batch_first: bool = False
    dropout: float = 0.
    bidirectional: bool = False
    proj_size: int = 0
    device: Any = None
    dtype: Any = None


args_list = list(LSTMConstructorArgs.model_fields.keys())
args_idx = {_arg: _idx for _idx, _arg in enumerate(args_list)}


def get_param(key, args, kwargs):
    if len(args) > args_idx[key]:
        return True, args[args_idx[key]]

    if key in kwargs:
        return True, kwargs[key]

    return False, None


def get_and_replace_param(key, func_value, args, kwargs):
    if len(args) > args_idx[key]:
        old_value = args[args_idx[key]]
        new_args = (
            args[:args_idx[key]] 
            + (func_value(old_value), ) 
            + args[args_idx[key]+1:])
        return (True, old_value), new_args, kwargs

    if key in kwargs:
        old_value = kwargs[key]
        new_kwargs = {k: v for k, v in kwargs.items() if k != key}
        new_kwargs[key] = func_value(old_value)
        return (True, old_value), args, new_kwargs

    kwargs[key] = func_value(None)
    return (False, None), args, kwargs


class ComplexSplitLSTM(nn.Module, ModuleInterface):
    ConstructorArgs = LSTMConstructorArgs

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        ((avail, batch_first), 
         new_args, 
         new_kwargs) = get_and_replace_param(
            'batch_first', lambda _: True, args, kwargs)
        if avail:
            assert batch_first == True, 'Batch first must be True!'

        self.hidden_size: int = get_param(
            'hidden_size', new_args, new_kwargs)

        self.lstm_r = nn.LSTM(*new_args, **new_kwargs)
        self.lstm_i = nn.LSTM(*new_args, **new_kwargs)

    
    def forward(self, x_b2td: torch.Tensor) -> torch.Tensor:
        batch_size, complex_dim, _, _ = x_b2td.size()
        assert complex_dim == 2, 'This axis must be 2D!'
        xri_btd = x_b2td.flatten(start_dim=0, end_dim=1)
        hrir_btd, _ = self.lstm_r(xri_btd)
        hrii_btd, _ = self.lstm_i(xri_btd)
        hr_b2td = hrir_btd.unflatten(
            dim=0, sizes=(batch_size, complex_dim))
        hi_b2td = hrii_btd.unflatten(
            dim=0, sizes=(batch_size, complex_dim))

        hrr_btd, hri_btd = hr_b2td[:, 0, :, :], hi_b2td[:, 0, :, :]
        hir_btd, hii_btd = hr_b2td[:, 1, :, :], hi_b2td[:, 1, :, :]

        hr_btd = hrr_btd - hii_btd
        hi_btd = hri_btd + hir_btd

        h_b2td = torch.stack((hr_btd, hi_btd), dim=1)

        return h_b2td


class ComplexFuseLSTM(nn.Module, ModuleInterface):
    ConstructorArgs = LSTMConstructorArgs

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        ((avail, batch_first), 
         new_args, 
         new_kwargs) = get_and_replace_param(
            'batch_first', lambda _: True, args, kwargs)
        if avail:
            assert batch_first == True, 'Batch first must be True!'

        _, new_args, new_kwargs = get_and_replace_param(
            'hidden_size', lambda v: 2*v, new_args, new_kwargs)
        self.hidden_size: int = get_param(
            'hidden_size', new_args, new_kwargs)

        self.lstm_ri = nn.LSTM(*new_args, **new_kwargs)

    def forward(self, x_b2td: torch.Tensor) -> torch.Tensor:
        batch_size, complex_dim, _, _ = x_b2td.size()
        assert complex_dim == 2, 'This axis must be 2D!'
        xri_btd = x_b2td.flatten(start_dim=0, end_dim=1)
        hriri_btd, _ = self.lstm_ri(xri_btd)
        hri_b2td = hriri_btd.unflatten(
            dim=0, sizes=(batch_size, complex_dim))

        hrr_btd, hri_btd = torch.chunk(
            hri_b2td[:, 0, :, :], chunks=2, dim=-1)
        hir_btd, hii_btd = torch.chunk(
            hri_b2td[:, 1, :, :], chunks=2, dim=-1)

        hr_btd = hrr_btd - hii_btd
        hi_btd = hri_btd + hir_btd

        h_b2td = torch.stack((hr_btd, hi_btd), dim=1)

        return h_b2td


def sanity_check():
    from deep.utils import count_parameters
    net_1 = ComplexSplitLSTM(input_size=4, hidden_size=8)
    net_2 = ComplexFuseLSTM(input_size=4, hidden_size=8)

    x_b2td = torch.randn(16, 2, 37, 4)
    y1_b2td = net_1(x_b2td)
    y2_b2td = net_2(x_b2td)

    print(x_b2td.size(), y1_b2td.size(), y2_b2td.size())
    print(count_parameters(net_1), count_parameters(net_2))


if __name__ == '__main__':
    sanity_check()
