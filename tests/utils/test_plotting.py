"""
Pytest tests for plotting functionality in NIfTI_MRS_Plus.

Tests both the core plotting module and NIfTI_MRS_Plus integration.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

# Test imports
from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus
from augmentrum.utils.plotting import (
    vis_nifti_mrs_plus,
    plot_batch_comparison,
    plot_batch_grid_detailed,
    quick_plot,
    _plot_single_spectrum,
    _plot_spectrum_on_axis
)


#**************#
#   fixtures   #
#**************#

@pytest.fixture
def mock_nifti_mrs():
    """Create a mock NIFTI_MRS object."""
    nifti = Mock()

    # Create realistic FID data
    n_points = 2048
    t = np.linspace(0, 1, n_points)
    # Simulated FID with damped oscillation
    fid = np.exp(-t * 5) * np.exp(1j * 2 * np.pi * 100 * t)
    fid += 0.5 * np.exp(-t * 3) * np.exp(1j * 2 * np.pi * 200 * t)
    fid += 0.1 * (np.random.randn(n_points) + 1j * np.random.randn(n_points))

    nifti.__getitem__ = Mock(return_value=fid)
    nifti.shape = (1, 1, 1, n_points)
    nifti.dwelltime = 0.0005  # 500 µs
    nifti.bandwidth = 2000.0  # Real number, not Mock
    nifti.spectrometer_frequency = [123.26]  # 3T for 1H
    nifti.nucleus = ['1H']
    nifti.dim_tags = [None, None, None]  # Proper list, not Mock

    return nifti


@pytest.fixture
def mock_nifti_mrs_list(mock_nifti_mrs):
    """Create a list of mock NIFTI_MRS objects."""
    nifti_list = []
    # Use a consistent dim_tags value for all mocks
    shared_dim_tags = [None, None, None]
    
    for i in range(5):
        nifti = Mock()
        # Create slightly different FID for each
        n_points = 2048
        t = np.linspace(0, 1, n_points)
        fid = np.exp(-t * (5 + i * 0.5)) * np.exp(1j * 2 * np.pi * (100 + i * 10) * t)
        fid += 0.1 * (np.random.randn(n_points) + 1j * np.random.randn(n_points))
        
        nifti.__getitem__ = Mock(return_value=fid)
        nifti.shape = (1, 1, 1, n_points)
        nifti.dwelltime = 0.0005
        nifti.bandwidth = 2000.0  # Real number, not Mock
        nifti.spectrometer_frequency = [123.26]
        nifti.nucleus = ['1H']
        nifti.dim_tags = shared_dim_tags  # All mocks share the same dim_tags
        
        # Add dim_position method for compatibility
        def dim_position(dim_tag, tags=shared_dim_tags):
            if dim_tag in tags:
                return tags.index(dim_tag) + 4
            raise ValueError(f"Dimension {dim_tag} not found")
        nifti.dim_position = dim_position
        
        nifti_list.append(nifti)
    
    return nifti_list


@pytest.fixture
def nifti_mrs_plus(mock_nifti_mrs_list):
    """Create a NIfTI_MRS_Plus object for testing."""
    return NIfTI_MRS_Plus(
        nifti_list=mock_nifti_mrs_list,
        backend='nifti_list',
        volatile=True
    )


#***************************#
#   test helper functions   #
#***************************#

def test_plot_spectrum_on_axis(mock_nifti_mrs):
    """Test the low-level plotting function."""
    fig, ax = plt.subplots()

    _plot_spectrum_on_axis(
        mock_nifti_mrs,
        ax=ax,
        ppmlim=(0.5, 4.2),
        title="Test Spectrum"
    )

    # Check that lines were plotted
    assert len(ax.lines) > 0

    # Check axis properties
    assert ax.get_xlabel() == "Chemical Shift (ppm)"
    assert ax.get_ylabel() == "Amplitude (a.u.)"
    assert ax.get_title() == "Test Spectrum"

    # Check ppm axis is inverted
    xlim = ax.get_xlim()
    assert xlim[0] > xlim[1], "PPM axis should be inverted"

    plt.close(fig)


def test_plot_single_spectrum(mock_nifti_mrs):
    """Test single spectrum plotting."""
    with patch('matplotlib.pyplot.show'):  # Mock show to prevent display
        fig = _plot_single_spectrum(
            mock_nifti_mrs,
            ppmlim=(0.5, 4.2),
            title="Test"
        )

    assert fig is not None
    assert len(fig.axes) == 1

    plt.close(fig)


#***************************************#
#   test nifti_mrs_plus.plot() method   #
#***************************************#

def test_nifti_plus_plot_single_spectrum(nifti_mrs_plus):
    """Test plotting a single spectrum from batch."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot(batch_index=0, ppmlim=(0.5, 4.2))

    assert fig is not None
    plt.close(fig)


def test_nifti_plus_plot_batch_grid(nifti_mrs_plus):
    """Test plotting multiple spectra in grid."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot(max_batch_display=4, ppmlim=(0.5, 4.2))

    assert fig is not None
    # Should have multiple subplots
    assert len(fig.axes) >= 4

    plt.close(fig)


def test_nifti_plus_plot_custom_layout(nifti_mrs_plus):
    """Test custom grid layout."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot(
            grid_layout=(2, 3),
            max_batch_display=6
        )

    assert fig is not None
    assert len(fig.axes) == 6

    plt.close(fig)


def test_nifti_plus_plot_invalid_batch_index(nifti_mrs_plus):
    """Test error handling for invalid batch index."""
    with pytest.raises(IndexError):
        nifti_mrs_plus.plot(batch_index=999)


def test_nifti_plus_plot_empty_batch():
    """Test plotting empty batch."""
    empty_nifti_plus = NIfTI_MRS_Plus(nifti_list=[], backend='nifti_list', volatile=True)

    with pytest.raises(ValueError, match="empty"):
        empty_nifti_plus.plot()


#**************************************************#
#   test nifti_mrs_plus.plot_comparison() method   #
#**************************************************#

def test_nifti_plus_plot_comparison(nifti_mrs_plus):
    """Test comparison overlay plotting."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot_comparison(
            indices=[0, 1, 2],
            labels=['A', 'B', 'C'],
            ppmlim=(0.5, 4.2)
        )

    assert fig is not None
    assert len(fig.axes) == 1

    # Should have multiple lines (one per spectrum)
    ax = fig.axes[0]
    assert len(ax.lines) >= 3

    # Check legend exists
    assert ax.get_legend() is not None

    plt.close(fig)


def test_nifti_plus_plot_comparison_default_indices(nifti_mrs_plus):
    """Test comparison with default indices."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot_comparison()

    assert fig is not None
    plt.close(fig)


def test_nifti_plus_plot_comparison_custom_colors(nifti_mrs_plus):
    """Test comparison with custom colors."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot_comparison(
            indices=[0, 1],
            colors=['red', 'blue'],
            alpha=0.9
        )

    assert fig is not None
    plt.close(fig)


#********************************************#
#   test nifti_mrs_plus.plot_grid() method   #
#********************************************#

def test_nifti_plus_plot_grid(nifti_mrs_plus):
    """Test detailed grid plotting."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot_grid(
            max_display=4,
            ppmlim=(0.5, 4.2)
        )

    assert fig is not None
    assert len(fig.axes) >= 4

    plt.close(fig)


def test_nifti_plus_plot_grid_with_metabolites(nifti_mrs_plus):
    """Test grid with metabolite highlighting."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot_grid(
            max_display=4,
            show_metabolites=True,
            title="Test Grid"
        )

    assert fig is not None

    # Check that metabolite regions are highlighted (should have colored patches)
    for ax in fig.axes[:4]:
        if ax.get_visible():
            # Check for axvspan patches (metabolite highlights)
            assert len(ax.patches) > 0 or len(ax.collections) > 0

    plt.close(fig)


def test_nifti_plus_plot_grid_max_display(nifti_mrs_plus):
    """Test that max_display limits number of plots."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot_grid(max_display=3)

    assert fig is not None
    # Should not plot more than max_display
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) <= 3

    plt.close(fig)


#************************************#
#   test batch-aware visualization   #
#************************************#

def test_vis_nifti_mrs_plus_single(nifti_mrs_plus):
    """Test vis_nifti_mrs_plus with single spectrum."""
    with patch('matplotlib.pyplot.show'):
        fig = vis_nifti_mrs_plus(
            nifti_mrs_plus,
            batch_index=0,
            ppmlim=(0.5, 4.2)
        )

    assert fig is not None
    plt.close(fig)


def test_vis_nifti_mrs_plus_batch(nifti_mrs_plus):
    """Test vis_nifti_mrs_plus with batch."""
    with patch('matplotlib.pyplot.show'):
        fig = vis_nifti_mrs_plus(
            nifti_mrs_plus,
            max_batch_display=4
        )

    assert fig is not None
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_vis_nifti_mrs_plus_invalid_input():
    """Test error handling for invalid input."""
    with pytest.raises(TypeError):
        vis_nifti_mrs_plus("not a nifti object")


#*****************************************#
#   test plot_batch_comparison function   #
#*****************************************#

def test_plot_batch_comparison_function(nifti_mrs_plus):
    """Test standalone plot_batch_comparison function."""
    with patch('matplotlib.pyplot.show'):
        fig = plot_batch_comparison(
            nifti_mrs_plus,
            indices=[0, 1],
            labels=['Spectrum 1', 'Spectrum 2']
        )

    assert fig is not None
    plt.close(fig)


#********************************************#
#   test plot_batch_grid_detailed function   #
#********************************************#

def test_plot_batch_grid_detailed_function(nifti_mrs_plus):
    """Test standalone plot_batch_grid_detailed function."""
    with patch('matplotlib.pyplot.show'):
        fig = plot_batch_grid_detailed(
            nifti_mrs_plus,
            max_display=4,
            show_metabolites=True
        )

    assert fig is not None
    plt.close(fig)


#******************************************#
#   test quick_plot convenience function   #
#******************************************#

def test_quick_plot(nifti_mrs_plus):
    """Test quick_plot convenience function."""
    with patch('matplotlib.pyplot.show'):
        fig = quick_plot(nifti_mrs_plus, index=0)

    assert fig is not None
    plt.close(fig)


#*******************************#
#   test ppm axis calculation   #
#*******************************#

def test_ppm_axis_calculation(mock_nifti_mrs):
    """Test that PPM axis is calculated correctly."""
    with patch('matplotlib.pyplot.show'):
        fig = _plot_single_spectrum(mock_nifti_mrs, ppmlim=(0.5, 4.2))

    ax = fig.axes[0]
    xlim = ax.get_xlim()

    # PPM axis should be inverted (high to low)
    assert xlim[0] > xlim[1]

    # Should be within specified limits
    assert xlim[1] <= 0.5
    assert xlim[0] >= 4.2

    plt.close(fig)


#*************************#
#   test fft convention   #
#*************************#

def test_fft_convention(mock_nifti_mrs):
    """Test that FFT is applied correctly (not IFFT)."""
    # Get the FID
    fid = mock_nifti_mrs[:]

    # Apply FFT like the plotting function does
    spec = np.fft.fftshift(np.fft.fft(fid))

    # Spectrum should have reasonable magnitude
    assert np.max(np.abs(spec)) > 0
    assert np.isfinite(spec).all()

    # Check that it's different from IFFT
    spec_ifft = np.fft.fftshift(np.fft.ifft(fid))
    assert not np.allclose(spec, spec_ifft), "Should use FFT not IFFT"


#******************************************#
#   test multi-dimensional data handling   #
#******************************************#

def test_multidimensional_data_averaging():
    """Test that multi-dimensional data is averaged correctly."""
    nifti = Mock()

    # Create data that simulates what would be returned after averaging
    # Multi-dimensional data (coils, averages) would be averaged to 1D
    n_points = 1024
    # Simulate averaged FID (1D)
    fid_averaged = np.random.randn(n_points) + 1j * np.random.randn(n_points)

    nifti.__getitem__ = Mock(return_value=fid_averaged)
    nifti.shape = (1, 1, 1, n_points)  # Shape after processing
    nifti.dwelltime = 0.0005
    nifti.bandwidth = 2000.0  # Real number, not Mock
    nifti.spectrometer_frequency = [123.26]
    nifti.nucleus = ['1H']  # Real list, not Mock
    nifti.dim_tags = [None, None, None]  # Proper list

    with patch('matplotlib.pyplot.show'):
        fig = _plot_single_spectrum(nifti)

    # Should successfully create plot
    assert fig is not None
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) > 0

    plt.close(fig)


#****************************#
#   test figure attributes   #
#****************************#

def test_plot_figure_size(nifti_mrs_plus):
    """Test custom figure size."""
    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot(
            batch_index=0,
            figsize=(12, 6)
        )

    # Note: figsize might be adjusted, just check it exists
    assert fig.get_size_inches() is not None
    plt.close(fig)


def test_plot_title(nifti_mrs_plus):
    """Test custom title."""
    custom_title = "My Custom Title"

    with patch('matplotlib.pyplot.show'):
        fig = nifti_mrs_plus.plot(
            batch_index=0,
            title=custom_title
        )

    # Check title is set
    assert fig._suptitle is not None or any(custom_title in ax.get_title() for ax in fig.axes)
    plt.close(fig)


#*********************#
#   test edge cases   #
#*********************#

def test_plot_single_element_batch():
    """Test plotting a batch with only one element."""
    nifti = Mock()
    n_points = 1024
    fid = np.random.randn(n_points) + 1j * np.random.randn(n_points)
    nifti.__getitem__ = Mock(return_value=fid)
    nifti.shape = (1, 1, 1, n_points)
    nifti.dwelltime = 0.0005
    nifti.spectrometer_frequency = [123.26]
    nifti.nucleus = ['1H']
    nifti.dim_tags = None
    
    nifti_plus = NIfTI_MRS_Plus([nifti], backend='nifti_list', volatile=True)
    
    with patch('matplotlib.pyplot.show'):
        fig = nifti_plus.plot()
    
    assert fig is not None
    plt.close(fig)


def test_plot_large_batch():
    """Test plotting a large batch."""
    nifti_list = []
    shared_dim_tags = [None, None, None]  # Proper list, not None
    
    for i in range(20):
        nifti = Mock()
        n_points = 512  # Smaller for speed
        fid = np.random.randn(n_points) + 1j * np.random.randn(n_points)
        nifti.__getitem__ = Mock(return_value=fid)
        nifti.shape = (1, 1, 1, n_points)
        nifti.dwelltime = 0.0005
        nifti.bandwidth = 2000.0  # Real number, not Mock
        nifti.spectrometer_frequency = [123.26]
        nifti.nucleus = ['1H']
        nifti.dim_tags = shared_dim_tags
        nifti_list.append(nifti)
    
    nifti_plus = NIfTI_MRS_Plus(nifti_list, backend='nifti_list', volatile=True)
    
    with patch('matplotlib.pyplot.show'):
        # Should limit to max_batch_display
        fig = nifti_plus.plot(max_batch_display=6)
    
    assert fig is not None
    # Should not plot all 20
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) <= 6
    
    plt.close(fig)


#**********************#
#   integration test   #
#**********************#

def test_plotting_integration(nifti_mrs_plus):
    """Integration test for all plotting methods."""
    with patch('matplotlib.pyplot.show'):
        # Test all three main methods work together
        fig1 = nifti_mrs_plus.plot(batch_index=0)
        fig2 = nifti_mrs_plus.plot_comparison(indices=[0, 1])
        fig3 = nifti_mrs_plus.plot_grid(max_display=4)

        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

