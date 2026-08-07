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

__version__ = "0.0.2"
__author__ = "John T. LaMaster, Julian P. Merkofer, Kay C. Igwe"
__email__ = "j.p.merkofer@tue.nl"

# Main API
#*************#
#   imports   #
#*************#
import nifti_mrs_plus

# Claim the NIfTI-MRS provenance record before anything can write one. The
# container is a shared transport, so left alone it would credit itself for
# work Augmentrum did.
nifti_mrs_plus.set_provenance("Augmentrum", __version__)

from augmentrum.core import Augmentrum, NIfTI_MRS_Plus, Backend, BaseModule, AugmentationPipeline  # noqa: E402

__all__ = [
    "__version__",
    "Augmentrum",
    "NIfTI_MRS_Plus",
    "Backend",
    "BaseModule",
    "AugmentationPipeline",
]

