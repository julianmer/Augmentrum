####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#          K. C. Igwe (kci2104@columbia.edu)                                                       #
#                                                                                                  #
# Created: 2025-10-25                                                                              #
#                                                                                                  #
# Purpose: Package entry point. Re-exports the public API: the Augmentrum container, the pipeline, #
#          and every augmentation module.                                                          #
#                                                                                                  #
####################################################################################################

"""
Augmentrum - A Data Augmentation Package for MR Spectroscopy

A modular Python framework designed to help researchers with limited in-vivo MRS data
create diverse, physically consistent datasets through flexible augmentation.
"""

__version__ = "0.0.2"
__author__ = "John T. LaMaster, Julian P. Merkofer, Kay C. Igwe"
__email__ = "j.p.merkofer@tue.nl"

# Main API
#*************#
#   imports   #
#*************#
from augmentrum.core import Augmentrum, NIfTI_MRS_Plus, Backend, BaseModule, AugmentationPipeline

__all__ = [
    "__version__",
    "Augmentrum",
    "NIfTI_MRS_Plus",
    "Backend",
    "BaseModule",
    "AugmentationPipeline",
]

