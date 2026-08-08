####################################################################################################
#                                       sampling/__init__.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-30                                                                              #
#                                                                                                  #
# Purpose: Re-exports the sampling modules, which cover how data is acquired: subject splits, coil #
#          and average draws, and k-space sampling.                                                #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from augmentrum.sampling.coil_sampling import (
    Birdcage,
    CoilSampler,
    MapSource,
    Supplied,
    T2starMove,
)
from augmentrum.sampling.dimension_sampling import AverageSampler, DimensionSampler
from augmentrum.sampling.subject_splitter import SubjectSplitter
from augmentrum.sampling.kspace_sampling import (
    KspaceGeometry,
    Trajectory,
    TrajectoryRegistry,
    ShotUndersampler,
    GridMask,
    ShotPerturbations,
    ShotIO,
    KspaceSampler,
    KspaceUndersampling,
)

__all__ = [
    'AverageSampler',
    'CoilSampler',
    'DimensionSampler',
    'Birdcage',
    'MapSource',
    'Supplied',
    'T2starMove',
    'SubjectSplitter',
    'KspaceReconstructor',
    'KspaceUndersampling',
    'KspaceGeometry',
    'Trajectory',
    'TrajectoryRegistry',
    'ShotUndersampler',
    'GridMask',
    'ShotPerturbations',
    'ShotIO',
    'KspaceSampler',
]


#*******************#
#   lazy exports    #
#*******************#

def __getattr__(name):
    """
    Import KspaceReconstructor only when it is asked for.

    It is the one part of Augmentrum that genuinely requires PyTorch, since it
    runs on torchkbnufft. Importing it eagerly would make "import augmentrum"
    pull in torch for everyone, including numpy-only users.
    """
    if name == "KspaceReconstructor":
        from augmentrum.sampling.kspace_reconstructor import KspaceReconstructor
        return KspaceReconstructor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
