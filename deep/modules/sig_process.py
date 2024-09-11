import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F


class PreEmphasis(nn.Module):
    class ConstructorArgs(BaseModel):
        alpha: float = 0.97

    def __init__(self, alpha: float = 0.97) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer(
            name='h_112',
            tensor=torch.tensor(
                [-self.alpha, 1.],
                dtype=torch.float32
            ).unsqueeze_(0).unsqueeze_(0))

    def forward(self, x_bt: torch.Tensor) -> torch.Tensor:
        x_b1t = x_bt.unsqueeze(1)
        x_b1t = F.pad(x_b1t, [1, 0], mode='reflect')
        return F.conv1d(x_b1t, self.h_112).squeeze(1)


def sanity_check():
    import numpy as np
    from matplotlib import pyplot as plt

    t = torch.arange(512, dtype=torch.float32) / 8000
    x_1t = (
        torch.cos(2 * torch.pi * 600 * t).unsqueeze_(0)
        + torch.cos(2 * torch.pi * 100 * t).unsqueeze_(0)
        + torch.cos(2 * torch.pi * 1000 * t).unsqueeze_(0)
        + torch.cos(2 * torch.pi * 2000 * t).unsqueeze_(0)
        + torch.cos(2 * torch.pi * 3000 * t).unsqueeze_(0)) / 5
    net = PreEmphasis()
    y_1t = net(x_1t)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x_1t.detach().cpu().numpy()[0, :])
    ax[0].plot(y_1t.detach().cpu().numpy()[0, :])

    X_f = np.fft.rfft(x_1t.detach().cpu().numpy()[0, :])
    Y_f = np.fft.rfft(y_1t.detach().cpu().numpy()[0, :])

    ax[1].plot(np.abs(X_f))
    ax[1].plot(np.abs(Y_f))

    fig.savefig('plot_debugs/preemphasis.png')
    plt.close(fig=fig)


if __name__ == '__main__':
    sanity_check()
