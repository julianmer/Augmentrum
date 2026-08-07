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

#*************#
#   imports   #
#*************#
from nifti_mrs_plus.ops import (
    fft, ifft, fftshift, ifftshift, to_numpy, match_backend,
    is_torch, is_jax, is_tf,
)
from augmentrum.utils.geometry import Affine

__all__ = [
    # tensor_ops
    'fft', 'ifft', 'fftshift', 'ifftshift', 'to_numpy', 'match_backend',
    'is_torch', 'is_jax', 'is_tf',
        # geometry
    'Affine',
]
