import torch
from typing import Any, Literal, Optional, Tuple, Union


def _unpack(t_X: Tuple[Any, ...]) -> Any:
    if len(t_X) == 0:
        return None
    elif len(t_X) == 1:
        return t_X[0]   # Unpacking
    else:
        return t_X


def _squeeze_dim(
        *list_X__1__: torch.Tensor,
        c_dim: int,
        keepdim: bool
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if keepdim:
        return _unpack(tuple(list_X__1__))

    return _unpack(tuple([
        X_i__1__.squeeze(dim=c_dim) 
        for X_i__1__ in list_X__1__]))


def r2c(ar____: torch.Tensor, ai____: Optional[torch.Tensor] = None, c_dim: int = 1) -> torch.Tensor:
    """
    Convert a real tensor to a complex tensor by appending zeros as imaginary parts.

    Args:
        ar____ (torch.Tensor): The input real tensor representing real part.
        ai____ (Optional[torch.Tensor]): The input real tensor representing imaginary part.
            Defaults to None.
        c_dim (int, optional): The dimension along which the complex numbers are added.
            Defaults to 1.

    Returns:
        torch.Tensor: The complex tensor with the real part from the input tensor and
            zeros as the imaginary part.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> r2c(a, -1)
        tensor([[[1., 0.], [2., 0.]], [[3., 0.], [4., 0.]]])
    """
    ar__1__ = ar____.unsqueeze(dim=c_dim)
    if ai____ is None:
        ai__1__ = torch.zeros_like(ar__1__)
    else:
        ai__1__ = ai____.unsqueeze(dim=c_dim)

    return torch.cat((ar__1__, ai__1__), dim=c_dim)


def p2c(ap___: torch.Tensor, c_dim: int = 1) -> torch.Tensor:
    """
    Converts unit polar coordinates (phase only) to complex numbers.

    Args:
        ap___ (torch.Tensor): Tensor containing phases.
        c_dim (int, optional): Dimension for concatenation. Defaults to 1.

    Returns:
        torch.Tensor: Tensor representing complex numbers in Cartesian form.

    Example:
    >>> ap = torch.tensor([0.0, math.pi / 2, math.pi])
    >>> result = p2c(ap, c_dim=1)
    >>> print(result)
    tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    """
    return r2c(ap___.cos(), ap___.sin(), c_dim=c_dim)



def mp2c(am___: torch.Tensor, ap___: torch.Tensor, c_dim: int = 1) -> torch.Tensor:
    """
    Converts polar coordinates (magnitude and phase) to complex numbers.

    Args:
        am___ (torch.Tensor): Tensor containing magnitudes.
        ap___ (torch.Tensor): Tensor containing phases.
        c_dim (int, optional): Dimension for concatenation. Defaults to 1.

    Returns:
        torch.Tensor: Tensor representing complex numbers in Cartesian form.

    Example:
    >>> am = torch.tensor([1.0, 2.0, 3.0])
    >>> ap = torch.tensor([0.0, math.pi / 2, math.pi])
    >>> result = mp2c(am, ap, c_dim=1)
    >>> print(result)
    tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])

    Notes:
    - The function computes the real and imaginary parts of complex numbers using the
      magnitudes and phases provided. It then concatenates the real and imaginary parts
      along the specified dimension to form complex numbers in Cartesian form.
    """
    return r2c(am___ * ap___.cos(), am___ * ap___.sin(), c_dim=c_dim)


def c2r(a__2__: torch.Tensor, c_dim: int = 1,
        keepdim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a complex tensor into its real and imaginary parts.

    Args:
        a__2__ (torch.Tensor): The input complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the real and imaginary parts.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> c2r(a)
        (tensor([[1.], [3.]]), tensor([[2.], [4.]]))
    """
    return _squeeze_dim(
        *a__2__.chunk(chunks=2, dim=c_dim),
        c_dim=c_dim, keepdim=keepdim)


def c_real(a__2__: torch.Tensor, c_dim: int = 1,
           keepdim: bool = True) -> torch.Tensor:
    """
    Extracts the real part of complex numbers.

    Args:
        a__2__ (torch.Tensor): Input complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Default is 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The result tensor.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> c_real(a)
        tensor([[1.0], [3.0]])
    """
    return _squeeze_dim(
        a__2__.narrow(dim=c_dim, start=0, length=1),
        c_dim=c_dim, keepdim=keepdim)


def c_imag(a__2__: torch.Tensor, c_dim: int = 1,
           keepdim: bool = True) -> torch.Tensor:
    """
    Extracts the imaginary part of complex numbers.

    Args:
        a__2__ (torch.Tensor): Input complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Default is 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The result tensor.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> c_real(a)
        tensor([[2.0], [4.0]])
    """
    return _squeeze_dim(
        a__2__.narrow(dim=c_dim, start=1, length=1),
        c_dim=c_dim, keepdim=keepdim)


def _c_mul(mode: Literal['r', 'i', 'c'], a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1) -> torch.Tensor:
    """
    Perform element-wise multiplication of complex tensors based on the specified mode.

    Args:
        mode (Literal['r', 'i', 'c']): The multiplication mode:
            - 'r': Real part multiplication only.
            - 'i': Imaginary part multiplication only.
            - 'c': Complex multiplication (both real and imaginary parts).
        a__2__ (torch.Tensor): The first complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The second complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.

    Returns:
        torch.Tensor: The result of element-wise complex multiplication based on the specified mode.
    """
    ar__1__, ai__1__ = c2r(a__2__=a__2__, c_dim=c_dim)
    br__1__, bi__1__ = c2r(a__2__=b__2__, c_dim=c_dim)

    out_components = []
    if mode in ['r', 'c']:
        out_components.append(ar__1__ * br__1__ - ai__1__ * bi__1__)
    if mode in ['i', 'c']:
        out_components.append(ai__1__ * br__1__ + ar__1__ * bi__1__)
    return torch.cat(out_components, dim=c_dim)


def c_mul(a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1) -> torch.Tensor:
    """
    Element-wise complex multiplication of two complex tensors.

    Args:
        a__2__ (torch.Tensor): The first complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The second complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.

    Returns:
        torch.Tensor: The result of complex multiplication, with shape matching the input tensors.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> b = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(2, 2)
        >>> c_mul(a, b)
        tensor([[ -7.,  16.], [-11.,  52.]])
    """
    return _c_mul(mode='c', a__2__=a__2__, b__2__=b__2__, c_dim=c_dim)
    

def c_mul_r(a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1,
            keepdim: bool = True) -> torch.Tensor:
    """
    Real part of the element-wise complex multiplication of two complex tensors.

    Args:
        a__2__ (torch.Tensor): The first complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The second complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The real part of the result of complex multiplication, with shape
            matching the input tensors.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> b = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(2, 2)
        >>> c_mul_r(a, b)
        tensor([[ -7.], [-11.]])
    """
    return _squeeze_dim(
        _c_mul(mode='r', a__2__=a__2__, b__2__=b__2__, c_dim=c_dim),
        c_dim=c_dim, keepdim=keepdim)
    

def c_mul_i(a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1,
            keepdim: bool = True) -> torch.Tensor:
    """
    Imaginary part of the element-wise complex multiplication of two complex tensors.

    Args:
        a__2__ (torch.Tensor): The first complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The second complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The imaginary part of the result of complex multiplication, with shape
            matching the input tensors.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> b = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(2, 2)
        >>> c_mul_r(a, b)
        tensor([[16.], [52.]])
    """
    return _squeeze_dim(
        _c_mul(mode='i', a__2__=a__2__, b__2__=b__2__, c_dim=c_dim),
        c_dim=c_dim, keepdim=keepdim)


def c_conj(a__2__: torch.Tensor, c_dim: int = 1) -> torch.Tensor:
    """
    Compute the complex conjugate of a complex tensor.

    Args:
        a__2__ (torch.Tensor): The input complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.

    Returns:
        torch.Tensor: The complex conjugate of the input tensor.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> c_conj(a)
        tensor([[ 1., -2.], [ 3., -4.]])
    """
    ar__1__, ai__1__ = c2r(a__2__=a__2__, c_dim=c_dim)
    return torch.cat((ar__1__, -ai__1__), dim=c_dim)


def c_square_mag(a__2__: torch.Tensor, c_dim: int = 1,
                 keepdim: bool = True) -> torch.Tensor:
    """
    Compute the square magnitude of complex numbers in a complex tensor.

    Args:
        a__2__ (torch.Tensor): The input complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The square magnitude of complex numbers in the input tensor.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> c_square_mag(a)
        tensor([[ 5.], [25.]])
    """
    return a__2__.square().sum(dim=c_dim, keepdim=keepdim)


def c_mag(a__2__: torch.Tensor, c_dim: int = 1,
          keepdim: bool = False) -> torch.Tensor:
    """
    Compute the magnitude of complex numbers in a complex tensor.

    Args:
        a__2__ (torch.Tensor): The input complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The magnitude of complex numbers in the input tensor.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> c_mag(a)
        tensor([[2.2361], [5.0000]])
    """
    return c_square_mag(a__2__, c_dim, keepdim).sqrt()


def c_phs(a__2__: torch.Tensor, c_dim: int = 1,
          keepdim: bool = True) -> torch.Tensor:
    """
    Compute the phase angles of complex numbers in a complex tensor.

    Args:
        a__2__ (torch.Tensor): The input complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The phase angles in radians of complex numbers in the input tensor.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> c_phs(a)
        tensor([[1.1071], [0.9273]])
    """
    ar__1__, ai__1__ = c2r(a__2__, c_dim=c_dim)
    return _squeeze_dim(
        torch.atan2(ai__1__, ar__1__),
        c_dim=c_dim, keepdim=keepdim)


def _c_div(mode: Literal['r', 'i', 'c'], a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1, eps: float = 0.0) -> torch.Tensor:
    """
    Perform element-wise division of complex tensors based on the specified mode.

    Args:
        mode (Literal['r', 'i', 'c']): The multiplication mode:
            - 'r': Real part division only.
            - 'i': Imaginary part division only.
            - 'c': Complex multiplication (both real and imaginary parts).
        a__2__ (torch.Tensor): The first complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The second complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.

    Returns:
        torch.Tensor: The result of element-wise complex division based on the specified mode.
    """
    return _c_mul(mode, a__2__, c_conj(b__2__, c_dim=c_dim), c_dim=c_dim) / c_square_mag(b__2__, c_dim=c_dim).add(eps)

    
def c_div(a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1, eps: float = 0.0) -> torch.Tensor:
    """
    Compute the element-wise division of two complex tensors.

    Args:
        a__2__ (torch.Tensor): The numerator complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The denominator complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.

    Returns:
        torch.Tensor: The result of element-wise complex division.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> b = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(2, 2)
        >>> c_div(a, b)
        tensor([[0.2787, 0.0656], [0.4690, 0.0354]])
    """
    return _c_div(mode='c', a__2__=a__2__, b__2__=b__2__, c_dim=c_dim, eps=eps)
    

def c_div_r(a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1,
            keepdim: bool = True, eps: float = 0.0) -> torch.Tensor:
    """
    Real part of the element-wise complex division of two complex tensors.

    Args:
        a__2__ (torch.Tensor): The first complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The second complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The real part of the result of complex division, with shape
            matching the input tensors.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> b = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(2, 2)
        >>> c_div_r(a, b)
        tensor([[0.2787], [0.4690]])
    """
    return _squeeze_dim(
        _c_div(mode='r', a__2__=a__2__, b__2__=b__2__, c_dim=c_dim, eps=eps),
        c_dim=c_dim, keepdim=keepdim)


def c_div_i(a__2__: torch.Tensor, b__2__: torch.Tensor, c_dim: int = 1,
            keepdim: bool = True, eps: float = 0.0) -> torch.Tensor:
    """ Imaginary part of the element-wise complex division of two complex tensors.
    Args:
        a__2__ (torch.Tensor): The first complex tensor with shape (..., 2, ...).
        b__2__ (torch.Tensor): The second complex tensor with shape (..., 2, ...).
        c_dim (int, optional): The dimension along which the complex numbers are stored.
            Defaults to 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not.
            Default to True.

    Returns:
        torch.Tensor: The imaginary part of the result of complex division, with shape
            matching the input tensors.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2)
        >>> b = torch.tensor([5.0, 6.0, 7.0, 8.0]).view(2, 2)
        >>> c_div_i(a, b)
        tensor([[0.0656], [0.0354]])
    """
    return _squeeze_dim(
        _c_div(mode='i', a__2__=a__2__, b__2__=b__2__, c_dim=c_dim, eps=eps),
        c_dim=c_dim, keepdim=keepdim)

def c_normalise(a__2__: torch.Tensor, c_dim: int = 1, 
                eps: float = 0.0) -> torch.Tensor:
    """
    Normalize a complex tensor along a specified dimension.

    This function normalizes a complex tensor by dividing it by the square root of 
    its magnitude squared along the specified dimension. An epsilon value can be added
    to the magnitude squared to avoid division by zero.

    Args:
        a__2__ (torch.Tensor): The input complex tensor to normalize. 
            This should be a tensor where the real and imaginary parts 
            are concatenated along the specified dimension.
        c_dim (int, optional): The dimension along which the normalization 
            is computed. Defaults to 1.
        eps (float, optional): A small value added to the magnitude squared 
            to prevent division by zero. Defaults to 0.0.

    Returns:
        torch.Tensor: The normalized complex tensor. The result has the 
            same shape as the input tensor.
    
    Examples:
        >>> import torch
        >>> a__2__ = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        >>> normalized_tensor = c_normalise(a__2__, c_dim=0)
        >>> print(normalized_tensor)
        tensor([[0.3162, 0.4472],
                [0.9487, 0.8944]])
    
    Notes:
        - The input tensor `a__2__` is expected to be a complex tensor, 
          with the real and imaginary components concatenated along the 
          specified dimension `c_dim`.
        - The function `c_square_mag` is assumed to calculate the square 
          of the magnitude of the complex tensor along the specified dimension.
        - The function uses `add(eps).sqrt()` to add a small epsilon value 
          to the magnitude squared before taking the square root, 
          which helps to avoid division by zero in cases where the magnitude 
          could be very small or zero.
    """
    return a__2__ / c_square_mag(a__2__, c_dim=c_dim, keepdim=True).add(eps).sqrt()
