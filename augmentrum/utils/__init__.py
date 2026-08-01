####################################################################################################
#                                        utils/__init__.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-03-29                                                                              #
#                                                                                                  #
# Purpose: Re-exports the shared utilities: tensor dispatch, array-constructor backends, geometry  #
#          helpers and plotting.                                                                   #
#                                                                                                  #
####################################################################################################

"""
Utility modules for Augmentrum.

Three concerns live here and are exported side by side:

``tensor_ops``
    Dispatch on the type of a tensor you already hold — FFT helpers, numpy
    conversion, cross-backend promotion.

``backends``
    Construct arrays when there is no input tensor to dispatch from, e.g. when
    generating k-space coordinates. ``ArrayBackend`` is the contract.

``geometry``
    Quaternion and affine matrix construction, shared by the spatial
    augmentations and anything else needing a resampling transform.
"""

#*************#
#   imports   #
#*************#
from augmentrum.utils.tensor_ops import (
    fft, ifft, fftshift, ifftshift, to_numpy, match_backend,
    is_torch, is_jax, is_tf,
)
from augmentrum.utils.backends import (
    ArrayBackend, NumpyBackend, TorchBackend, validate_backend,
)
from augmentrum.utils.geometry import Affine

__all__ = [
    # tensor_ops
    'fft', 'ifft', 'fftshift', 'ifftshift', 'to_numpy', 'match_backend',
    'is_torch', 'is_jax', 'is_tf',
    # backends
    'ArrayBackend', 'NumpyBackend', 'TorchBackend', 'validate_backend',
    # geometry
    'Affine',
]
