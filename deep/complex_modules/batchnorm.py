import torch
from torch import nn
from typing import Optional, Any
from deep.modules.utils import inverse_square_root_2x2
from deep.base_module import ModuleInterface


# NOTE: Currently this implementation does not apply LoRA to 
# affine transforms of normalisation layers.
class ComplexBatchNorm(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        num_features: int
        eps: float = 1e-5
        momentum: Optional[float] = 0.1
        affine: bool = True
        track_running_stats: bool = True
        device: Any = None
        dtype: Any = None

    def __init__(self, *args, **kwargs) -> None:
        ModuleInterface.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        num_features = self._configs.num_features
        eps = self._configs.eps
        momentum = self._configs.momentum
        affine = self._configs.affine
        track_running_stats = self._configs.track_running_stats 
        device= self._configs.device
        dtype = self._configs.dtype

        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: Optional[float] = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats

        n_feats = self.num_features

        if self.affine:
            self.Wrr_c = nn.Parameter(
                data=torch.empty(
                    n_feats, device=device, dtype=dtype),
                requires_grad=True)
            self.Wri_c = nn.Parameter(
                data=torch.empty(
                    n_feats, device=device, dtype=dtype),
                requires_grad=True)
            self.Wii_c = nn.Parameter(
                data=torch.empty(
                    n_feats, device=device, dtype=dtype),
                requires_grad=True)
            self.Br_c = nn.Parameter(
                data=torch.empty(
                    n_feats, device=device, dtype=dtype),
                requires_grad=True)
            self.Bi_c = nn.Parameter(
                data=torch.empty(
                    n_feats, device=device, dtype=dtype),
                requires_grad=True)
        else:
            self.register_parameter('Wrr_c', None)
            self.register_parameter('Wri_c', None)
            self.register_parameter('Wii_c', None)
            self.register_parameter('Br_c', None)
            self.register_parameter('Bi_c', None)

        if self.track_running_stats:
            self.register_buffer(
                name='RMr_c',
                tensor=torch.zeros(n_feats, dtype=torch.float32))
            self.register_buffer(
                name='RMi_c',
                tensor=torch.zeros(n_feats, dtype=torch.float32))
            self.register_buffer(
                name='RVrr_c', 
                tensor=torch.ones(n_feats, dtype=torch.float32))
            self.register_buffer(
                name='RVri_c', 
                tensor=torch.zeros(n_feats, dtype=torch.float32))
            self.register_buffer(
                name='RVii_c', 
                tensor=torch.ones(n_feats, dtype=torch.float32))
            self.register_buffer(
                name='num_batches_tracked',
                tensor=torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr_c', None)
            self.register_parameter('RMi_c', None)
            self.register_parameter('RVrr_c', None)
            self.register_parameter('RVri_c', None)
            self.register_parameter('RVii_c', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr_c.zero_()
            self.RMi_c.zero_()
            self.RVrr_c.fill_(1)
            self.RVri_c.zero_()
            self.RVii_c.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br_c.data.zero_()
            self.Bi_c.data.zero_()

            # W will be positive-definite
            self.Wrr_c.data.fill_(1)
            self.Wri_c.data.uniform_(-.9, +.9)
            self.Wii_c.data.fill_(1)

    def forward(self, x_b2c__: torch.Tensor) -> torch.Tensor:
        _, complex_dim, in_features = x_b2c__.size()[:3]
        assert in_features == self.num_features
        assert complex_dim == 2

        xr_bc__ = x_b2c__[:, 0, ...]
        xi_bc__ = x_b2c__[:, 1, ...]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1

            # use cumulative moving average
            if self.momentum is None:
                exponential_average_factor = (
                    1.0 / self.num_batches_tracked.item())
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # NOTE: The precise meaning of the "training flag" is:
        # - True:  Normalize using batch statistics, update 
        # running statistics if they are being collected.
        # - False: Normalize using running statistics, ignore 
        # batch statistics.
        training = self.training or not self.track_running_stats

        # Dimension list except feature dimension
        redux = [
            i for i in reversed(range(xr_bc__.dim())) if i != 1]

        # Shape of averaged vector
        vdim = [1] * xr_bc__.dim()
        vdim[1] = xr_bc__.size(1)

        # Mean M Computation and Centering
        # Includes running mean update if training and running.
        if training:
            Mr_1c__ = xr_bc__.mean(dim=redux, keepdim=True)
            Mi_1c__ = xi_bc__.mean(dim=redux, keepdim=True)
            if self.track_running_stats:
                Mr_c = Mr_1c__.squeeze(dim=redux)
                Mi_c = Mi_1c__.squeeze(dim=redux)
                # NOTE: For torch < 2.0
                # Mr_c = Mr_1c__
                # Mi_c = Mi_1c__
                # for d in redux:
                #     Mr_c = Mr_c.squeeze(dim=d)
                #     Mi_c = Mi_c.squeeze(dim=d)
                # RM += (exp_avg_factor * Mr)
                self.RMr_c.lerp_(
                    Mr_c.detach().float(), exponential_average_factor)
                self.RMi_c.lerp_(
                    Mi_c.detach().float(), exponential_average_factor)
        else:
            Mr_1c__ = self.RMr_c.view(vdim)
            Mi_1c__ = self.RMi_c.view(vdim)
        xr_bc__, xi_bc__ = xr_bc__ - Mr_1c__, xi_bc__ - Mi_1c__

        # Variance Matrix V Computation
        # Includes epsilon 
        #   numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        if training:
            Vrr_bc__ = xr_bc__ * xr_bc__
            Vri_bc__ = xr_bc__ * xi_bc__
            Vii_bc__ = xi_bc__ * xi_bc__

            Vrr_1c__ = Vrr_bc__.mean(dim=redux, keepdim=True)
            Vri_1c__ = Vri_bc__.mean(dim=redux, keepdim=True)
            Vii_1c__ = Vii_bc__.mean(dim=redux, keepdim=True)
            if self.track_running_stats:
                Vrr_c = Vrr_1c__.squeeze(dim=redux)
                Vri_c = Vri_1c__.squeeze(dim=redux)
                Vii_c = Vii_1c__.squeeze(dim=redux)

                # NOTE: For torch < 2.0
                # Vrr_c = Vrr_1c__
                # Vri_c = Vri_1c__
                # Vii_c = Vii_1c__
                # for d in redux:
                #     Vrr_c = Vrr_c.squeeze(dim=d)
                #     Vii_c = Vii_c.squeeze(dim=d)
                #     Vri_c = Vri_c.squeeze(dim=d)

                self.RVrr_c.lerp_(
                    Vrr_c.detach().float(), exponential_average_factor)
                self.RVri_c.lerp_(
                    Vri_c.detach().float(), exponential_average_factor)
                self.RVii_c.lerp_(
                    Vii_c.detach().float(), exponential_average_factor)
        else:
            Vrr_1c__ = self.RVrr_c.view(vdim)
            Vri_1c__ = self.RVri_c.view(vdim)
            Vii_1c__ = self.RVii_c.view(vdim)

        Vrr_1c__ = Vrr_1c__ + self.eps
        Vri_1c__ = Vri_1c__
        Vii_1c__ = Vii_1c__ + self.eps

        Urr_1c__, Uri_1c__, Uii_1c__ = inverse_square_root_2x2(
            Vrr_1c__, Vri_1c__, Vii_1c__)

        # Optionally left-multiply U by affine weights W to produce 
        # combined weights Z, left-multiply the inputs by Z, then 
        # optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr_1c__ = self.Wrr_c.view(vdim)
            Wri_1c__ = self.Wri_c.view(vdim)
            Wii_1c__ = self.Wii_c.view(vdim)
            Zrr_1c__ = Wrr_1c__*Urr_1c__ + Wri_1c__*Uri_1c__
            Zri_1c__ = Wrr_1c__*Uri_1c__ + Wri_1c__*Uii_1c__
            Zir_1c__ = Wri_1c__*Urr_1c__ + Wii_1c__*Uri_1c__
            Zii_1c__ = Wri_1c__*Uri_1c__ + Wii_1c__*Uii_1c__
        else:
            Zrr_1c__, Zri_1c__, Zir_1c__, Zii_1c__ = (
                Urr_1c__, Uri_1c__, Uri_1c__, Uii_1c__)

        yr_bc__ = Zrr_1c__*xr_bc__ + Zri_1c__*xi_bc__
        yi_bc__ = Zir_1c__*xr_bc__ + Zii_1c__*xi_bc__

        if self.affine:
            yr_bc__ = yr_bc__ + self.Br_c.view(vdim)
            yi_bc__ = yi_bc__ + self.Bi_c.view(vdim)

        y_b2c__ = torch.stack([yr_bc__, yi_bc__], dim=1)
        return y_b2c__
        

def sanity_check_complex_bn():
    net = ComplexBatchNorm(4)

    net.train()
    for _ in range(1000):
        x_bct = 2 * torch.randn(16, 2, 4, 32) + 1
        y_bct = net(x_bct)

    print(net.RMr_c)
    print(net.RMi_c)
    print(net.RVrr_c)
    print(net.RVri_c)
    print(net.RVii_c)

    net.eval()
    with torch.no_grad():
        x_bct = torch.randn(16, 2, 4, 32)
        y_bct = net(x_bct)
        print(y_bct.size())


if __name__ == '__main__':
    sanity_check_complex_bn()
