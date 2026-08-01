####################################################################################################
#                                     augmentation/__init__.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-13                                                                              #
#                                                                                                  #
# Purpose: Re-exports the augmentation modules, each a BaseModule that perturbs MRS data in a      #
#          physically motivated way.                                                               #
#                                                                                                  #
####################################################################################################

"""
Augmentation modules for MRS data.
"""

#*************#
#   imports   #
#*************#
from augmentrum.augmentation.gaussian_noise import GaussianNoise
from augmentrum.augmentation.line_broadening import LineBroadening
from augmentrum.augmentation.baseline_augmentation import BaselineAugmentation
from augmentrum.augmentation.residual_water import ResidualWater
from augmentrum.augmentation.spurious_echoes import SpuriousEchoes
from augmentrum.augmentation.artificial_peaks import ArtificialPeaks
from augmentrum.augmentation.eddy_current import EddyCurrent
from augmentrum.augmentation.apodization import Apodization
from augmentrum.augmentation.phase_frequency import PhaseShift, FrequencyShift
from augmentrum.augmentation.amplitude_scaling import AmplitudeScaling
from augmentrum.augmentation.spatial_augmentations import SpatialAugmentations
from augmentrum.augmentation.zero_fill import ZeroFill

__all__ = [
    'GaussianNoise',
    'LineBroadening',
    'BaselineAugmentation',
    'ResidualWater',
    'SpuriousEchoes',
    'ArtificialPeaks',
    'EddyCurrent',
    'Apodization',
    'PhaseShift',
    'FrequencyShift',
    'AmplitudeScaling',
    'SpatialAugmentations',
    'ZeroFill',
]
