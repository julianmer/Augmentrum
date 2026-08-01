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

"""
Sampling modules for Augmentrum.

Covers how data is *acquired*: which subjects land in which split, which coils and
averages are drawn, and which parts of k-space are measured.

``KspaceReconstructor`` is re-exported, but it needs the optional ``torchkbnufft``
dependency to *run*. That import is deferred to first use, so importing this
package never depends on it.
"""

#*************#
#   imports   #
#*************#
from augmentrum.sampling.coil_average_sampler import CoilAverageSampler
from augmentrum.sampling.subject_splitter import SubjectSplitter
from augmentrum.sampling.kspace_reconstructor import KspaceReconstructor
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
    'CoilAverageSampler',
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
