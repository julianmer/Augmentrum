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

#*************#
#   imports   #
#*************#
from augmentrum.processing.domain import Domain, DomainError, DomainTransform
from augmentrum.processing.raw_processing import RawProcessor

__all__ = [
    'Domain',
    'DomainError',
    'DomainTransform',
    'RawProcessor',
    'HermiteMAkimaInterpolator',
    'BicubicHermiteMAkima2D',
    'TricubicHermiteMAkima3D',
]


#******************#
#   lazy exports   #
#******************#
_INTERPOLATORS = ("HermiteMAkimaInterpolator", "BicubicHermiteMAkima2D",
                  "TricubicHermiteMAkima3D")


def __getattr__(name):
    """
    Import the Hermite interpolators only when they are asked for.

    They are PyTorch modules and nothing in the augmentation pipeline uses them,
    so importing them eagerly would make "import augmentrum" pull in torch for
    everyone.
    """
    if name in _INTERPOLATORS:
        from augmentrum.processing import interpolating
        return getattr(interpolating, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
