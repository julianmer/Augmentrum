####################################################################################################
#                                         core/__init__.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Re-exports the core data structures: the NIfTI_MRS_Plus container, the module base      #
#          class, and the pipeline.                                                                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from augmentrum.core.augmentrum import Augmentrum
from augmentrum.core.base_module import BaseModule
from augmentrum.core.pipeline import AugmentationPipeline

__all__ = ['NIfTI_MRS_Plus', 'Backend', 'Augmentrum', 'BaseModule', 'AugmentationPipeline']
