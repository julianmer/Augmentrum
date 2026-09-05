####################################################################################################
#                                        physics/__init__.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-05                                                                              #
#                                                                                                  #
# Purpose: Physics building blocks that are not themselves pipeline stages - operators and phase   #
#          terms that augmentation modules compose, mirroring how "sampling" holds both            #
#          BaseModule stages and plain operators like KspaceReconstructor/GriddingNUFFT.            #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from augmentrum.physics.concomitant_field import (
    CONCOMITANT_BASIS_TERMS,
    ConcomitantFieldPhase,
    concomitant_field_coefficients,
    concomitant_field_basis,
)

__all__ = [
    'ConcomitantFieldPhase',
    'concomitant_field_coefficients',
    'concomitant_field_basis',
    'CONCOMITANT_BASIS_TERMS',
]
