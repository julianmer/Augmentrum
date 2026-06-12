"""
Augmentrum - A Data Augmentation Package for MR Spectroscopy

A modular Python framework designed to help researchers with limited in-vivo MRS data
create diverse, physically consistent datasets through flexible augmentation.
"""

__version__ = "0.0.2"
__author__ = "John T. LaMaster, Julian P. Merkofer, Kay C. Igwe"
__email__ = "j.p.merkofer@tue.nl"

# Main API
from augmentrum.core import Augmentrum, NIfTI_MRS_Plus, Backend, BaseModule, AugmentationPipeline

__all__ = [
    "__version__",
    "Augmentrum",
    "NIfTI_MRS_Plus",
    "Backend",
    "BaseModule",
    "AugmentationPipeline",
]

