####################################################################################################
#                                      processing/__init__.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-30                                                                              #
#                                                                                                  #
# Purpose: Re-exports the processing modules, which cover what happens to data after acquisition:  #
#          raw preprocessing and off-grid interpolation.                                           #
#                                                                                                  #
####################################################################################################

"""
Processing modules for Augmentrum.

Covers what happens to data *after* acquisition:

``nifti_raw_processor``
    FSL-MRS raw preprocessing — coil combination, alignment, averaging, eddy
    current correction, water removal, phase correction.

``interpolating``
    Hermite modified-Akima resampling of gridded volumes at off-grid
    coordinates, i.e. evaluating a volume along a non-Cartesian trajectory.

``utils``
    MRS helper functions shared by the above.
"""

#*************#
#   imports   #
#*************#
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.processing.interpolating import (
    HermiteMAkimaInterpolator, BicubicHermiteMAkima2D, TricubicHermiteMAkima3D,
)

__all__ = [
    'NIfTI_RawProcessor',
    'HermiteMAkimaInterpolator',
    'BicubicHermiteMAkima2D',
    'TricubicHermiteMAkima3D',
]
