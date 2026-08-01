####################################################################################################
#                                        tensor_ops.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-03-27                                                                              #
#                                                                                                  #
# Purpose: Backend-agnostic tensor operations for MRS signal processing. Detects the array         #
#          library from the tensor type and dispatches to numpy / torch / jax transparently,       #
#          so callers write one expression that works with any tensor type.                        #
#                                                                                                  #
# The dispatch happens *once* per call inside these helpers — callers never                        #
# need to check tensor types themselves.                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np


#***********************#
#   backend detection   #
#***********************#

def is_torch(x):
    """True if *x* is a PyTorch tensor."""
    return type(x).__module__.split(".")[0] == "torch"


def is_jax(x):
    """True if *x* is a JAX array."""
    return type(x).__module__.split(".")[0] in ("jax", "jaxlib")


def is_tf(x):
    """True if *x* is a TensorFlow tensor."""
    return type(x).__module__.split(".")[0] == "tensorflow"


#************************#
#   complex fft / ifft   #
#************************#
# All three backends (numpy, torch >= 1.7, jax) support native complex-valued
# FFT with the same semantics, so we just dispatch to the right module.
# No need for the keras (real, imag)-tuple workaround.

def fft(x):
    """Complex-to-complex FFT along the last axis (any backend).

    MRS convention: ``fft`` maps *spectrum -> FID*.

    Args:
        x: Complex tensor of any shape.

    Returns:
        Complex tensor with the same shape as *x*.
    """
    if is_torch(x):
        import torch
        return torch.fft.fft(x)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.fft(x)
    return np.fft.fft(x)


def ifft(x):
    """Complex-to-complex inverse FFT along the last axis (any backend).

    MRS convention: ``ifft`` maps *FID -> spectrum*.

    Args:
        x: Complex tensor of any shape.

    Returns:
        Complex tensor with the same shape as *x*.
    """
    if is_torch(x):
        import torch
        return torch.fft.ifft(x)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.ifft(x)
    return np.fft.ifft(x)


#**************************#
#   fftshift / ifftshift   #
#**************************#
# numpy / jax use ``axes=`` keyword; torch uses ``dim=``.

def fftshift(x, axis=-1):
    """Shift zero-frequency component to the centre (like ``np.fft.fftshift``).

    Args:
        x: Tensor of any shape.
        axis: Axis (or axes) along which to shift.

    Returns:
        Shifted tensor.
    """
    if is_torch(x):
        import torch
        return torch.fft.fftshift(x, dim=axis)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.fftshift(x, axes=axis)
    return np.fft.fftshift(x, axes=axis)


def ifftshift(x, axis=-1):
    """Inverse of :func:`fftshift` (like ``np.fft.ifftshift``).

    Args:
        x: Tensor of any shape.
        axis: Axis (or axes) along which to shift.

    Returns:
        Shifted tensor.
    """
    if is_torch(x):
        import torch
        return torch.fft.ifftshift(x, dim=axis)
    if is_jax(x):
        import jax.numpy as jnp
        return jnp.fft.ifftshift(x, axes=axis)
    return np.fft.ifftshift(x, axes=axis)


#************************#
#   conversion helpers   #
#************************#

def to_numpy(x):
    """Convert *any* backend tensor to a NumPy ``ndarray``.

    Handles PyTorch ``.detach().cpu()`` and JAX/TF device arrays
    transparently.

    Args:
        x: Tensor or array.

    Returns:
        ``numpy.ndarray``
    """
    if isinstance(x, np.ndarray):
        return x
    if is_torch(x):
        return x.detach().cpu().numpy()
    # JAX, TF, and anything with __array__ protocol
    return np.asarray(x)


def match_backend(param, ref):
    """Convert a NumPy parameter array to the same backend as *ref*.

    This is the key helper for the "auto-promote" pattern.  Augmentation
    modules generate envelopes / noise / phase ramps as lightweight NumPy
    arrays, then call::

        result = data_tensor * match_backend(envelope_np, data_tensor)

    This avoids the known issue where ``torch.Tensor * np.ndarray`` can
    fail when ``requires_grad=True`` (numpy tries to call ``__array__``
    on the grad-enabled tensor via ufunc dispatch).

    For NumPy-to-NumPy this is a no-op (zero cost).

    Args:
        param: Parameter array (typically numpy).
        ref:   Reference tensor whose backend we want to match.

    Returns:
        ``param`` converted to the same framework as ``ref``.
    """
    if isinstance(param, np.ndarray):
        if isinstance(ref, np.ndarray):
            return param                             # no-op: numpy handles mixed dtypes
        if is_torch(ref):
            import torch
            return torch.as_tensor(param, device=ref.device)
        if is_jax(ref):
            import jax.numpy as jnp
            return jnp.array(param)
        if is_tf(ref):
            import tensorflow as tf
            # TF is strict: complex64 tensor * complex128 constant raises.
            # Cast param to match ref's dtype before wrapping.
            return tf.constant(param, dtype=ref.dtype)
    # Already same backend, or unknown → return as-is
    return param
