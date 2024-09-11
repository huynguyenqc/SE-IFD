import math
import random
import torch
from torch import nn, distributions
from torch.nn import functional as F
from typing import List, Optional, Tuple

from deep.base_module import ModuleInterface


class ReservoirSampler(nn.Module):
    """ 
    Reservoir sampler on batch data based on L algorithm

    Reference: 
        [1] https://en.wikipedia.org/wiki/Reservoir_sampling
        [2] Kim-Hung Li. 1994. Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n))). 
            ACM Trans. Math. Softw. 20, 4 (Dec. 1994), 481–493. https://doi.org/10.1145/198429.198435
    """
    def __init__(self, n_reservoir_samples: int, embedding_dim: int, dtype: Optional[torch.dtype] = None) -> None:
        super(ReservoirSampler, self).__init__()
        self.n_reservoir_samples: int = n_reservoir_samples
        self.embedding_dim: int = embedding_dim

        # Number of observed samples
        self.register_buffer('n_observed_samples', torch.tensor(0, dtype=torch.int64))

        # The most recent sample index to be chosen to the reservoir
        self.register_buffer('current_index', torch.tensor(0, dtype=torch.int64))

        # For algorithm L
        self.register_buffer('w_gen', torch.tensor(1.0, dtype=torch.float))

        # Reservoir samples
        self.register_buffer('r_ld', torch.empty((self.n_reservoir_samples, embedding_dim), dtype=dtype))

    def reset_reservoir(self) -> None:
        self.n_observed_samples.fill_(0)
        self.current_index.fill_(0)
        self.w_gen.fill_(1.0)

    @staticmethod
    def u() -> float:
        """Random a float in (0, 1)

        Returns:
            float: Output number
        """
        eps = 1e-6
        return min(max(random.random(), eps), 1.0 - eps)

    @property
    def collect_enough(self) -> bool:
        return self.current_index.item() >= self.n_reservoir_samples

    def update_from_data(self, x_nd: torch.Tensor) -> None:
        with torch.no_grad():
            n = x_nd.size(0)
            previous_n_observed_samples = self.n_observed_samples.item()
            self.n_observed_samples.add_(n)

            # If the current number of reservoir samples are not enough
            if self.current_index.item() < self.n_reservoir_samples:
                cur_idx = self.current_index.item()
                assert cur_idx == previous_n_observed_samples

                # Use as much observed samples as possible for reservoir
                Q = min(self.n_observed_samples.item(), self.n_reservoir_samples) - cur_idx

                self.r_ld[cur_idx: cur_idx + Q, :].copy_(x_nd[: Q, ...])
                self.current_index.add_(Q)

            # Implementation trick: Force to choose the immediate next sample to the reservoir
            if self.current_index.item() == self.n_reservoir_samples:
                self.current_index.add_(1)

            # If the number of reservoir samples are enough already
            if self.current_index.item() > self.n_reservoir_samples:
                # Use dict to keep the latest replacements only
                candidate_dict = dict()

                while self.current_index.item() <= self.n_observed_samples.item():
                    candidate_idx = self.current_index.item() - previous_n_observed_samples - 1
                    updated_idx = random.randrange(self.n_reservoir_samples)
                    candidate_dict[updated_idx] = candidate_idx

                    self.w_gen.mul_(math.exp(math.log(self.u()) / self.n_reservoir_samples))
                    self.w_gen.clamp_(min=1e-6, max=1.0 - 1e-6)
                    self.current_index.add_(math.floor(math.log(self.u()) / math.log(1.0 - self.w_gen.item())) + 1)

                if len(candidate_dict) > 0:
                    candidate_indices = []
                    updated_indices = []
                    for updated_idx, candidate_idx in candidate_dict.items():
                        candidate_indices.append(candidate_idx)
                        updated_indices.append(updated_idx)

                    i_n = torch.tensor(data=candidate_indices, device=x_nd.device)
                    l_n = torch.tensor(data=updated_indices, device=x_nd.device)

                    self.r_ld.index_copy_(dim=0, index=l_n, source=x_nd.index_select(dim=0, index=i_n).float())

    def get_reservoir_samples(self) -> torch.Tensor:
        return self.r_ld.clone()


class Codebook(nn.Module, ModuleInterface):
    class ConstructorArgs(ModuleInterface.ConstructorArgs):
        dim_codebook: int
        codebook_size: int

    def __init__(self, **kwargs) -> None:
        ModuleInterface.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.embeddings = nn.Embedding(
            num_embeddings=self._configs.codebook_size,
            embedding_dim=self._configs.dim_codebook)

        init_range = 1 / self._configs.codebook_size
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self._eps: float = torch.finfo(torch.float32).eps

    @property
    def dim_codebook(self) -> int:
        return self.embeddings.embedding_dim

    @property
    def codebook_size(self) -> int:
        return self.embeddings.num_embeddings

    @staticmethod
    def pairwise_distance(
            u_nd: torch.Tensor, v_md: torch.Tensor) -> torch.Tensor:
        v_dm = v_md.t()
        d_nm = torch.addmm(
            input=(u_nd.square().sum(-1, keepdim=True) 
                   + v_dm.square().sum(0, keepdim=True)),
            mat1=u_nd, mat2=v_dm, beta=1, alpha=-2)
        
        return d_nm

    @staticmethod
    def perplexity(oneHot__k: torch.Tensor) -> torch.Tensor:
        oneHot_nk = oneHot__k.flatten(start_dim=0, end_dim=-2)
        pSelected_k = oneHot_nk.float().mean(dim=0)
        entropy = -torch.sum(pSelected_k * torch.log(pSelected_k + 1e-10))
        results = torch.exp(entropy)
        return results

    def lookup(self, idx__: torch.Tensor) -> torch.Tensor:
        return self.embeddings(idx__)

    def forward(self, x__d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x__d (torch.Tensor): Input vectors
        Return:
            xq__d (torch.Tensor): Corresponding vectors to the samples
            p__k (torch.Tensor): Estimated posterior distribution
        """
        input_size = x__d.size()[: -1]  # Except last dimension
        x_nd = x__d.flatten(start_dim=0, end_dim=-2)
        
        d_nk = self.pairwise_distance(x_nd, self.embeddings.weight)
        idx_n = torch.argmin(d_nk, dim=-1)
        xq_nd = self.lookup(idx_n)
        oneHot_nk = F.one_hot(idx_n, num_classes=self.codebook_size).to(x_nd.dtype)

        xq__d = xq_nd.unflatten(dim=0, sizes=input_size)
        p__k = oneHot_nk.unflatten(dim=0, sizes=input_size)

        return xq__d, p__k


class EMACodebook(Codebook):
    class ConstructorArgs(Codebook.ConstructorArgs):
        gamma: float = 0.99
        epsilon: float = 1e-5
        n_reservoir_samples: Optional[int] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.gamma: float = self._configs.gamma
        self.epsilon: float = self._configs.epsilon

        # Turn off `requires_grad` for embedding weight -> no update by optimiser
        self.embeddings.weight.requires_grad_(False)

        # EMA cluster size
        self.register_buffer('N_k', torch.ones(self._configs.codebook_size))
        # EMA running sum
        self.register_buffer('m_kd', self.embeddings.weight.data.clone().requires_grad_(False))

        # Reservoir sampling
        if self._configs.n_reservoir_samples is not None:
            assert self._configs.n_reservoir_samples > self._configs.codebook_size, \
                'Number of reservoir samples must be larger than codebook size!'

            self.reservoir_sampler = ReservoirSampler(
                n_reservoir_samples=self._configs.n_reservoir_samples,
                embedding_dim=self._configs.dim_codebook,
                dtype=self.embeddings.weight.dtype)
        else:
            self.reservoir_sampler = None

    def set_codebook_ema_momentum(self, lr: Optional[float] = None) -> None:
        """Update momentum of EMA method from model's learning rate

        Args:
            lr (Optional[float], optional): Model learning rate. Defaults to None.
        """
        # if lr is not None:
        #     # Codebook learning rate is set to 10 times larger than model learning rate
        #     codebook_lr = 100. * lr   

        #     if codebook_lr < 0.5:
        #         self.gamma = 1. - 2. * codebook_lr
        pass

    def update_reservoir(self, x__d: torch.Tensor) -> None:
        if self.reservoir_sampler is not None and self.training:
            with torch.no_grad():
                x_nd = x__d.flatten(start_dim=0, end_dim=-2).detach()
                self.reservoir_sampler.update_from_data(x_nd=x_nd)
               
    def initialise_codebook_from_reservoir(self) -> None:
        """ K-Mean++ algorithm from reservoir samples """
        if self.reservoir_sampler is not None and self.training:
            # Must collect enough reservoir samples
            assert self.reservoir_sampler.collect_enough, 'Reservoir sampler must collect enough samples!'
            with torch.no_grad():
                L = self.reservoir_sampler.n_reservoir_samples  # Number of unused reservoir samples
                K = self.codebook_size
                r_ld = self.reservoir_sampler.get_reservoir_samples()

                # Available reservoir samples
                iR_l = torch.tensor(data=list(range(L)), dtype=torch.int, device=self.embeddings.weight.device)

                # Selected centre samples
                iC_k = torch.zeros(K, dtype=torch.int, device=self.embeddings.weight.device)

                for k in range(K):
                    if k == 0:
                        idx = random.randrange(L)
                    else:
                        # Pairwise distance between current centres and unused reservoir samples
                        d_l, _ = self.pairwise_distance(
                            r_ld.index_select(dim=0, index=iR_l[: L]),
                            r_ld.index_select(dim=0, index=iC_k[: k])
                        ).min(dim=-1)

                        # Probability (weight) of selecting is proportional to the (squared) distance
                        idx = random.choices(population=range(L), weights=d_l.tolist(), k=1)[0]

                    # Add selected index to the centre samples
                    iC_k[k] = iR_l[idx]

                    # Remove selected index from the reservoir samples
                    iR_l[idx] = iR_l[L-1]
                    L -= 1

                self.embeddings.weight.data.copy_(r_ld.index_select(dim=0, index=iC_k))

                self.N_k.fill_(1.0)
                self.m_kd.copy_(self.embeddings.weight.data)

                self.reservoir_sampler.reset_reservoir()

    def forward(self, x__d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x__d (torch.Tensor): Input vectors
        Return:
            xq__d (torch.Tensor): Corresponding vectors to the samples
            p__k (torch.Tennsor): Posterior distribution
        """
        input_size = x__d.size()[: -1]  # Except last dimension
        x_nd = x__d.flatten(start_dim=0, end_dim=-2)
        
        d_nk = self.pairwise_distance(x_nd, self.embeddings.weight)
        idx_n = torch.argmin(d_nk, dim=-1)
        oneHot_nk = F.one_hot(idx_n, num_classes=self.codebook_size).to(x__d.dtype)
        p__k = oneHot_nk.unflatten(dim=0, sizes=input_size)

        # Codebook update using EMA
        if self.training:
            with torch.no_grad():
                # Update cluster size using EMA
                n_k = oneHot_nk.sum(dim=0)
                self.N_k.mul_(self.gamma).add_((1 - self.gamma) * n_k)
                
                # Laplace smoothing to avoid empty clusters
                N_1 = self.N_k.sum()
                self.N_k.add_(self.epsilon).div_(N_1 + self.codebook_size * self.epsilon).mul_(N_1)

                # Update running sum
                dm_kd = oneHot_nk.t() @ x_nd
                self.m_kd.mul_(self.gamma).add_((1 - self.gamma) * dm_kd)

            # Update codebook
            self.embeddings.weight.data.copy_(self.m_kd / self.N_k.unsqueeze(-1))

        # The selected codes are from updated codebook (the centroids of the batch)
        xq_nd = self.lookup(idx_n)
        xq__d = xq_nd.unflatten(dim=0, sizes=input_size)

        return xq__d, p__k


class GumbelCodebook(Codebook):
    class ConstructorArgs(Codebook.ConstructorArgs):
        tau: float = 0.5

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tau: float = self._configs.tau

    def forward(self, x__d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x__d (torch.Tensor): Input vectors
        Return:
            xq__d (torch.Tensor): Corresponding vectors to the samples
            p__k (torch.Tennsor): Posterior distribution
        """
        input_size = x__d.size()[: -1]  # Except last dimension
        x_nd = x__d.flatten(start_dim=0, end_dim=-2)

        d_nk = self.pairwise_distance(x_nd, self.embeddings.weight)

        gumbel_softmax = distributions.RelaxedOneHotCategorical(
            logits=-d_nk, temperature=self.tau)

        p_nk = gumbel_softmax.probs
        p__k = p_nk.unflatten(dim=0, sizes=input_size)

        oneHotSoft_nk = gumbel_softmax.rsample()

        if self.training:
            xq_nd = oneHotSoft_nk @ self.embeddings.weight
        else:
            xq_nd = self.lookup(torch.argmax(oneHotSoft_nk, dim=-1))

        xq__d = xq_nd.unflatten(dim=0, sizes=input_size)

        return xq__d, p__k


def sanity_check():
    import numpy as np
    from torch import optim
    from torch.nn import functional as F

    np.random.seed(0)
    torch.manual_seed(0)

    train_data_nd = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=np.float32)
    test_data_nd = np.array([[0, 0], [12, 3]], dtype=np.float32)

    init_embeddings = torch.empty(2, 2).uniform_(-0.5, 0.5)

    vq = Codebook(dim_codebook=2, codebook_size=2)
    vq_ema = EMACodebook(dim_codebook=2, codebook_size=2, gamma=0.5)
    vq_gumbel = GumbelCodebook(dim_codebook=2, codebook_size=2, tau=0.1)

    vq.embeddings.weight.data.copy_(init_embeddings)
    vq_ema.embeddings.weight.data.copy_(init_embeddings)
    vq_gumbel.embeddings.weight.data.copy_(init_embeddings)

    vq_optim = optim.Adam(vq.parameters(), lr=1.0)
    vq_gumbel_optim = optim.Adam(vq_gumbel.parameters(), lr=1.0)

    vq.train()
    vq_ema.train()
    vq_gumbel.train()

    x_nd = torch.from_numpy(train_data_nd)

    print('-- VQVAE --')
    centroid_1_traj, centroid_2_traj = [tuple(init_embeddings.numpy()[0].tolist())], [tuple(init_embeddings.numpy()[1].tolist())]
    for epoch in range(10):
        print(f'# Epoch: {epoch}')

        # Train VQ
        vq_optim.zero_grad()
        xQ_nd, p_nk = vq(x_nd.clone())
        vq_loss = F.mse_loss(xQ_nd, x_nd.detach())
        vq_loss.backward()
        vq_optim.step()
        print(f'    VQ loss: {vq_loss.item():.4f}')
        print(f'    Probabilities: {p_nk.detach().cpu().numpy().tolist()}')
        centroid_1_traj.append(tuple(vq.embeddings.weight.detach().cpu().numpy()[0].tolist()))
        centroid_2_traj.append(tuple(vq.embeddings.weight.detach().cpu().numpy()[1].tolist()))
    print(tuple(centroid_1_traj))
    print(tuple(centroid_2_traj))

    print('-- VQVAE EMA --')
    centroid_1_traj, centroid_2_traj = [tuple(init_embeddings.numpy()[0].tolist())], [tuple(init_embeddings.numpy()[1].tolist())]
    for epoch in range(10):
        print(f'# Epoch: {epoch}')

        # Train VQ-EMA
        xQ_nd, p_nk = vq_ema(x_nd.clone())
        vq_loss = F.mse_loss(xQ_nd, x_nd.detach())
        print(f'    VQ loss: {vq_loss.item():.4f}')
        print(f'    Probabilities: {p_nk.detach().cpu().numpy().tolist()}')
        centroid_1_traj.append(tuple(vq_ema.embeddings.weight.detach().cpu().numpy()[0].tolist()))
        centroid_2_traj.append(tuple(vq_ema.embeddings.weight.detach().cpu().numpy()[1].tolist()))
    print(tuple(centroid_1_traj))
    print(tuple(centroid_2_traj))

    print('-- VQ Gumbel --')
    centroid_1_traj, centroid_2_traj = [tuple(init_embeddings.numpy()[0].tolist())], [tuple(init_embeddings.numpy()[1].tolist())]
    for epoch in range(10):
        print(f'# Epoch: {epoch}')

        # Train VQ-Gumbel
        vq_gumbel_optim.zero_grad()
        xQ_nd, p_nk = vq_gumbel(x_nd.clone())
        dirac_nk = F.one_hot(p_nk.argmax(dim=-1), num_classes=2).float().detach()
        vq_loss = ((-dirac_nk * (p_nk + 1e-12).log()).sum(dim=-1)).mean()  # Maximise log prob.
        vq_loss.backward()
        vq_gumbel_optim.step()

        xQ_nd, p_nk = vq_gumbel(x_nd.clone())
        vq_loss = F.mse_loss(xQ_nd, x_nd.detach())
        print(f'    VQ loss: {vq_loss.item():.4f}')
        print(f'    Probabilities: {p_nk.detach().cpu().numpy().tolist()}')
        centroid_1_traj.append(tuple(vq_gumbel.embeddings.weight.detach().cpu().numpy()[0].tolist()))
        centroid_2_traj.append(tuple(vq_gumbel.embeddings.weight.detach().cpu().numpy()[1].tolist()))
    print(tuple(centroid_1_traj))
    print(tuple(centroid_2_traj))


if __name__ == '__main__':
    sanity_check()
