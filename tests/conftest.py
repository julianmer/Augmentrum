"""
Pytest configuration and shared fixtures for Augmentrum tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from augmentrum.core import NIfTI_MRS_Plus, Backend


@pytest.fixture
def dummy_nifti_mrs():
    """
    Create a single dummy NIfTI-MRS object with proper dimensions.

    Shape: (1, 1, 1, 2048, 8, 16)
    - x, y, z: 1, 1, 1 (spatial)
    - time: 2048 (FID points)
    - coils: 8
    - averages: 16
    """
    data = np.random.randn(1, 1, 1, 2048, 8, 16) + 1j * np.random.randn(1, 1, 1, 2048, 8, 16)
    nifti = gen_nifti_mrs(data, 1/2000, 123.0)
    nifti.set_dim_tag(4, 'DIM_COIL')
    nifti.set_dim_tag(5, 'DIM_DYN')
    return nifti


@pytest.fixture
def dummy_nifti_list(dummy_nifti_mrs):
    """Create a list of 5 dummy NIfTI-MRS objects."""
    nifti_list = []
    for i in range(5):
        data = np.random.randn(1, 1, 1, 2048, 8, 16) + 1j * np.random.randn(1, 1, 1, 2048, 8, 16)
        nifti = gen_nifti_mrs(data, 1/2000, 123.0)
        nifti.set_dim_tag(4, 'DIM_COIL')
        nifti.set_dim_tag(5, 'DIM_DYN')
        nifti_list.append(nifti)
    return nifti_list


@pytest.fixture
def nifti_mrs_plus(dummy_nifti_list):
    """Create a NIfTI_MRS_Plus object from dummy list."""
    return NIfTI_MRS_Plus(nifti_list=dummy_nifti_list, backend=Backend.NIFTI_LIST)


@pytest.fixture
def dummy_nifti_single_coil():
    """Create a dummy NIfTI-MRS object with single coil and no dynamics."""
    data = np.random.randn(1, 1, 1, 2048) + 1j * np.random.randn(1, 1, 1, 2048)
    nifti = gen_nifti_mrs(data, dwelltime=1/2000, spec_freq=123.0)
    return nifti


@pytest.fixture
def dummy_nifti_water():
    """Create a dummy water reference NIfTI-MRS object."""
    data = np.random.randn(1, 1, 1, 2048, 8, 16) + 1j * np.random.randn(1, 1, 1, 2048, 8, 16)
    nifti = gen_nifti_mrs(data, dwelltime=1/2000, spec_freq=123.0)
    nifti.set_dim_tag(4, 'DIM_COIL')
    nifti.set_dim_tag(5, 'DIM_DYN')
    return nifti

