import math
import random
import torch
from torch.nn import functional as F
from typing import Literal, Tuple
from deep.complex_modules import complex_op as c_op


def ceil_div(a: int, b: int) -> int:
    return (a-1) // b + 1


def compute_pad_len_for_inference(
        x__t: torch.Tensor,
        stft_win_len: int,
        stft_hop_len: int,
        required_stride: int = 1,
        look_ahead_frames: int = 0) -> int:
    """ Compute causal padding length required for inference

    Args:
        x__t (torch.Tensor): Input signal tensor
        stft_win_len (int): Window length
        stft_hop_len (int): Hop length
        required_stride (int, optional): The stride of the model. 
                                         Defaults to 1.
        look_ahead_frames (int, optional): Number of look-ahead frames. 
                                           Defaults to 0.

    Returns:
        int: Causal pad length
    """

    sig_len = x__t.size(-1)

    n_strided_frames_required = ceil_div(
        (look_ahead_frames - 1) * stft_hop_len + stft_win_len + sig_len,
        required_stride * stft_hop_len)
    n_frames_required = required_stride * n_strided_frames_required
    sig_len_required = (n_frames_required + 1) * stft_hop_len - stft_win_len
    causal_pad_required = sig_len_required - sig_len

    return causal_pad_required

    
def future_estimation(xHat__t: torch.Tensor, look_ahead_width: int = 0) -> torch.Tensor:
    """ Get future samples of a provided estimated signal, given look-ahead width """
    return xHat__t[..., look_ahead_width: ]

    
def past_groundtruth(x__t: torch.Tensor, xHat__t: torch.Tensor) -> torch.Tensor:
    """ Get past samples of a provided ground truth signal, given estimated one """
    out_len = xHat__t.size(-1)
    return x__t[..., : out_len]


def apply_augmentation(logXp_bft: torch.Tensor, SNRdB_range: Tuple[float, float] = (0, 35)) -> torch.Tensor:
    with torch.no_grad():
        SNRdB_b11 = logXp_bft.new_empty(
            (logXp_bft.size(0), 1, 1)
        ).uniform_(SNRdB_range[0], SNRdB_range[1])
        SNRs_b11 = 10 ** (SNRdB_b11 / 10)

        Xp_bft = logXp_bft.exp()

        # Speech power
        sigma2X_b11 = Xp_bft.mean(dim=[1, 2], keepdim=True)

        # Noise power
        sigma2N_b11 = sigma2X_b11 / SNRs_b11

        # Generate noise power spectrum
        G1_bft = torch.randn_like(logXp_bft).mul_(0.5 * sigma2N_b11.sqrt())
        G2_bft = torch.randn_like(logXp_bft).mul_(0.5 * sigma2N_b11.sqrt())
        Np_bft = G1_bft.square() + G2_bft.square()

        return (Np_bft + Xp_bft).log()


def apply_spectral_augmentation(logXp_bft: torch.Tensor) -> torch.Tensor:
    augmentation_rate = 0.1

    if random.random() > augmentation_rate:
        return logXp_bft

    with torch.no_grad():
        Bb, Ff, Tt = logXp_bft.size()
        neg_inf = -20 * math.log(10)

        n_freqs = 5
        n_times = 3

        ta_1t = torch.arange(n_times, device=logXp_bft.device).unsqueeze_(0)
        fa_1f = torch.arange(n_freqs, device=logXp_bft.device).unsqueeze_(0)
        tm_b1 = torch.randint(Tt-n_times+1, size=(Bb, 1), device=logXp_bft.device)
        fm_b1 = torch.randint(Ff-n_freqs+1, size=(Bb, 1), device=logXp_bft.device)

        mt_b1t = torch.ones(
            Bb, Tt, device=logXp_bft.device, dtype=logXp_bft.dtype
        ).scatter_(dim=1, index=ta_1t+tm_b1, value=neg_inf).unsqueeze_(dim=1)

        mf_bf1 = torch.ones(
            Bb, Ff, device=logXp_bft.device, dtype=logXp_bft.dtype
        ).scatter_(dim=1, index=fa_1f+fm_b1, value=neg_inf).unsqueeze_(dim=2)

        m_bft = mt_b1t * mf_bf1
        return logXp_bft * m_bft    # + neg_inf * (1. - m_bft)


def kl_divergence_log_normal_unit_variance(
        muA__: torch.Tensor,
        muB__: torch.Tensor) -> torch.Tensor:
    """KL divergence between two log normal distributions both of which have variance of 1

    Args:
        muA__ (torch.Tensor): Parameter of the first distribution
        muB__ (torch.Tensor): Parameter of the second distribution

    Returns:
        torch.Tensor: KL divergence between the two distributions
    """
    return 0.5 * F.mse_loss(input=muB__, target=muA__)


def kl_divergence_exponential(
        logSigma2A__: torch.Tensor,
        logSigma2B__: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two exponential distributions
        Exp(x; lambdaA) and Exp(x; lambdaB)
    
    Args:
        logSigma2A (torch.Tensor): Parameter of the first distribution (= log(1 / lambdaA))
        logSigma2B (torch.Tensor): Parameter of the first distribution (= log(1 / lambdaB))
    """
    r__ = logSigma2A__ - logSigma2B__
    return (r__.exp() - r__ - 1).mean()


def kl_divergence_categorical_with_mask(
        P__: torch.Tensor,
        Q__: torch.Tensor,
        m__: torch.Tensor,
        dim: int) -> torch.Tensor:
    """
    KL divergence between two categorical distribution

    Args:
        P__: First categorical distributions
        Q__: Second categorical distributions
        m__: Whether or not the distribution is involved
        dim: Dimension of category
    """
    eps = 1e-6
    return ((P__ * (P__ / (Q__ + eps) + eps).log()).sum(dim=dim) * m__).mean()

    
def kl_divergence_categorical(
        P__: torch.Tensor, 
        Q__: torch.Tensor, 
        dim: int) -> torch.Tensor:
    """
    KL divergence between two categorical distribution

    Args:
        P__ (torch.Tensor): First categorical distributions
        Q__ (torch.Tensor): Second categorical distributions
        dim (int): Dimension of category
    """
    eps = 1e-6
    return ((P__ * (P__ / (Q__ + eps) + eps).log()).sum(dim=dim)).mean()


def kl_divergence_normal(
        muA__: torch.Tensor,
        logSig2A__: torch.Tensor,
        muB__: torch.Tensor,
        logSig2B__: torch.Tensor,
        dim: int) -> torch.Tensor:
    """
    KL divergence between to multivariate normal distributions with diagonal covariance matrices

    Args:
        muA__ (torch.Tensor): Mean of the first distribution
        logSig2A__ (torch.Tensor): Diagon of the covariance matrix of the first distribution
        muB__ (torch.Tensor): Mean of the second distribution
        logSig2B__ (torch.Tensor): Diagon of the covariance matrix of the second distribution
        dim (int): The axis along which represents distribution vector dimension
    """

    logRatioAB__ = logSig2A__ - logSig2B__

    return (
        0.5 * (
            -logRatioAB__ - 1 
            + (muA__ - muB__).square().mul((-logSig2B__).exp()) 
            + logRatioAB__.exp()
        ).sum(dim=dim)
    ).mean()


def negative_signal_to_distortion_ratio_decibel(
        x__t: torch.Tensor,
        xHat__t: torch.Tensor
) -> torch.Tensor:
    """
    Negative signal-to-distortion ratio in decibel

    Args:
        x__t (torch.Tensor): Target (ground-truth) signal
        xHat__t (torch.Tensor): Estimated signal

    Returns:
        torch.Tensor: Negative signal-to-distortion ration (in dB)
    """
    eps = 1e-5

    alpha__1 = (xHat__t * x__t).mean(dim=-1, keepdim=True) / (
        x__t.square().mean(dim=-1, keepdim=True) + eps)

    x__t = alpha__1 * x__t
    sdr__ = x__t.square().mean(dim=-1) / (
        (xHat__t - x__t).square().mean(dim=-1) + eps)

    sdrdB__ = -10 * (sdr__ + eps).log10()
    return sdrdB__.mean()


def squared_euclidean_distance(
        x__d__: torch.Tensor,
        y__d__: torch.Tensor,
        dim: int) -> torch.Tensor:
    """
    Calculates the squared Euclidean distance between two tensors along a specified dimension.

    Args:
        x__d__ (torch.Tensor): Input tensor x with shape (..., dim, ...).
        y__d__ (torch.Tensor): Input tensor y with the same shape as x.
        dim (int): Dimension along which the squared Euclidean distance is calculated.

    Returns:
        torch.Tensor: Average of all squared Euclidean distance between x and y.
    """
    return (x__d__ - y__d__).square().sum(dim=dim).mean()


def complex_mask_multiplication(
        X_b2ft: torch.Tensor,
        M_b2ft: torch.Tensor) -> torch.Tensor:
    """ Complex mask multiplication """

    # Only keep the phase, normalise magnitude to 1
    M_b2ft = c_op.c_normalise(M_b2ft, c_dim=1, eps=1e-5)

    # Mask multiplication
    Y_b2ft = c_op.c_mul(X_b2ft, M_b2ft, c_dim=1)

    return Y_b2ft


def diff_keep_size(
        X__: torch.Tensor, 
        dim: int, 
        mode: Literal['backward', 'forward'] = 'backward') -> torch.Tensor:
    """
    Computes the discrete difference of a tensor along a specified dimension while keeping its size constant.

    Args:
        X__ (torch.Tensor): Input tensor.
        dim (int): Dimension along which the difference is computed.
        mode (Literal['backward', 'forward'], optional): Direction of difference. Defaults to 'backward'.
            - 'backward': Computes the difference using backward difference.
            - 'forward': Computes the difference using forward difference.

    Returns:
        torch.Tensor: Tensor containing the discrete difference of X__ along the specified dimension, 
            while keeping its size constant.

    Example:
    >>> X = torch.tensor([[1, 2, 4], [4, 7, 11]])
    >>> dim = 1
    >>> mode = 'backward'
    >>> result = diff_keep_size(X, dim, mode)
    >>> print(result)
    tensor([[1, 2], [3, 4]])

    Notes:
    - The function pads the tensor with zeros along the specified dimension based on the mode.
    - It then computes the difference of the padded tensor along the specified dimension.
    """
    assert -X__.dim() <= dim < X__.dim()

    # Inverse dim (for padding indexing)
    i_dim = (X__.dim() - 1 - dim) % X__.dim()
    pad_len = [0] * (2 * X__.dim())
    if mode == 'backward':
        pad_len[2*i_dim] = 1        # Add padding at the left
    elif mode == 'forward':
        pad_len[2*i_dim + 1] = 1    # Add padding at the right
    return torch.diff(F.pad(X__, pad=pad_len), dim=dim)   # Left derivative


def wrap(X__: torch.Tensor, T: float) -> torch.Tensor:
    """
    Wraps the values of a tensor within a specified period.

    Args:
        X__ (torch.Tensor): Input tensor containing values to be wrapped.
        T (float): Period for wrapping the values. Must be positive.

    Returns:
        torch.Tensor: Tensor with values wrapped within the specified period.

    Raises:
        AssertionError: If the period T is not positive.

    Example:
    >>> X = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> T = 2.0
    >>> result = wrap(X, T)
    >>> print(result)
    tensor([1.5000, 0.5000, 1.5000, 0.5000, 1.5000])

    Notes:
    - The function divides the input tensor values by the period T, floors the result,
      and then subtracts the integer multiples of T from the original values to wrap them
      within the range [0, T).
    """
    assert T > 0, 'The period must be positive!'

    k__ = X__.div(T).floor()
    return X__ - k__ * T


def wrap_between(X__: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """
    Wraps the values of a tensor within a specified range.

    Args:
        X__ (torch.Tensor): Input tensor containing values to be wrapped.
        vmin (float): Minimum value of the range.
        vmax (float): Maximum value of the range.

    Returns:
        torch.Tensor: Tensor with values wrapped within the specified range.

    Example:
    >>> X = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> vmin = 1.0
    >>> vmax = 3.0
    >>> result = wrap_between(X, vmin, vmax)
    >>> print(result)
    tensor([1.5000, 2.5000, 1.5000, 2.5000, 1.5000])

    Notes:
    - The function wraps the input tensor values within the specified range [vmin, vmax).
      It first normalizes the input values to the range [0, vmax - vmin), then wraps them
      using the wrap function, and finally adjusts them back to the original range [vmin, vmax).
    """
    return wrap(X__ - vmin, T=vmax - vmin) + vmin


def angle_mclaurin_series_expansion(
        phi__: torch.Tensor, dim: int, n_components: int) -> torch.Tensor:
    """
    Compute the McLaurin series expansion of the trigonometric functions (cosine and sine) 
    of a tensor of angles.

    Args:
        phi__: A tensor of angles. Must be within the range [-pi, pi] 
            for McLaurin expansion.
        dim (int): The dimension along which to perform expansion.
        n_components (int): The number of expansion components.

    Returns:
        torch.Tensor: A tensor representing the concatenated cosine and sine 
        expansion terms along the specified dimension. Each component tensor 
        has dimensions augmented by n_components along the specified dimension.
    """
    original_size = phi__.size()

    # Use last dimension as expansion dim
    cos_phi_d__ = [phi__.new_zeros(original_size) for _ in range(n_components)]
    sin_phi_d__ = [phi__.new_zeros(original_size) for _ in range(n_components)]

    # Expansion computation
    cos_phi_d__[0] += 1
    sin_phi_d__[0] += phi__

    phiSquare__ = phi__.square()

    for n in range(1, n_components):
        cos_phi_d__[n] = -cos_phi_d__[n-1] * (phiSquare__ / ((2 * n) * (2 * n - 1)))
        sin_phi_d__[n] = -sin_phi_d__[n-1] * (phiSquare__ / ((2 * n + 1) * (2 * n)))

    return torch.stack(cos_phi_d__ + sin_phi_d__, dim=dim)


def angle_multiplication(
        phiEmbedding__d__: torch.Tensor, dim: int, alpha__: torch.Tensor) -> torch.Tensor:
    """
    Compute McLaurin Series Expansion of an angle multiplied by a scaling value, given the 
    McLaurin series expansion of that angle.

    Args:
        phiEmbedding__d__ (torch.Tensor): A tensor containing angle embeddings 
            provided by McLaurin serires expansion.
        dim (int): The dimension along which to perform multiplication.
        alpha__ (torch.Tensor): A tensor of scalar coefficients. 
            Must be in the range [0, 1].

    Returns:
        torch.Tensor: Embedding (McLaurin serires expansion) of an multiplied angle.
    """
    n_components_twice = phiEmbedding__d__.size(dim)
    assert n_components_twice % 2 == 0
    n_components = n_components_twice // 2

    orderList = 2 * torch.arange(n_components, dtype=torch.int64, device=phiEmbedding__d__.device)
    powerList_d = torch.cat((orderList, orderList + 1))

    newSizePowerList = [1] * phiEmbedding__d__.dim()
    newSizePowerList[dim] = n_components_twice
    
    alphaEmbedding__d__= alpha__.unsqueeze(dim=dim).pow(powerList_d.view(newSizePowerList))

    return phiEmbedding__d__ * alphaEmbedding__d__


def angle_polar_from_mclaurin(phiEmbedding__d__: torch.Tensor, dim: int) -> torch.Tensor:
    n_dims = phiEmbedding__d__.dim()
    positive_dim = (dim + n_dims) % n_dims
    return phiEmbedding__d__.unflatten(dim=dim, sizes=(2, -1)).sum(dim=positive_dim+1)
