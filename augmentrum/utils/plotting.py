####################################################################################################
#                                           plotting.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-02-13                                                                              #
#                                                                                                  #
# Purpose: State-of-the-art plotting utilities for NIfTI_MRS_Plus objects                          #
#          Implements batch-aware plotting following NIfTI-MRS conventions                         #
#                                                                                                  #
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
import warnings

try:
    from fsl_mrs.utils.plotting import plot_spectrum, plot_spectra
    from fsl_mrs.core.mrs import MRS
    from fsl_mrs.utils import constants
    FSL_MRS_AVAILABLE = True
except ImportError:
    FSL_MRS_AVAILABLE = False
    warnings.warn(
        "FSL-MRS not available. Some plotting features will be limited. "
        "Install FSL-MRS for full functionality: pip install fsl-mrs"
    )


def vis_nifti_mrs_plus(
    data,
    display_dim=None,
    ppmlim=None,
    plot_avg=False,
    batch_index=None,
    max_batch_display=6,
    grid_layout=None,
    legend=True,
    figsize=None,
    title=None
):
    """
    Visualize NIfTI_MRS_Plus objects with batch-aware plotting.
    
    This function mirrors the NIfTI-MRS plot() method but adds batch support.
    
    Args:
        data: NIfTI_MRS_Plus object
        display_dim: Dimension to display (if multiple dims exist)
        ppmlim: Tuple of (min_ppm, max_ppm) for x-axis limits
        plot_avg: If True, plot average spectrum
        batch_index: Which batch element to plot (None = plot all up to max_batch_display)
        max_batch_display: Maximum number of batch elements to display
        grid_layout: Tuple (rows, cols) for grid layout. Auto if None
        legend: Whether to show legend
        figsize: Figure size (width, height)
        title: Plot title
        
    Returns:
        matplotlib figure object
        
    Example:
        >>> nifti_plus.plot()  # Plot first spectrum
        >>> nifti_plus.plot(batch_index=0)  # Plot first batch element
        >>> nifti_plus.plot(batch_index=None, max_batch_display=4)  # Plot first 4
        >>> nifti_plus.plot(ppmlim=(0.5, 4.2))  # Custom ppm range
    """
    from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus
    
    if not isinstance(data, NIfTI_MRS_Plus):
        raise TypeError("data must be a NIfTI_MRS_Plus object")
    
    # Get list of NIFTI_MRS objects
    nifti_list = data.list()
    
    if len(nifti_list) == 0:
        raise ValueError("NIfTI_MRS_Plus object is empty")
    
    # Determine ppm limits
    if ppmlim is None and len(nifti_list) > 0:
        first_nifti = nifti_list[0]
        if hasattr(first_nifti, 'nucleus') and FSL_MRS_AVAILABLE:
            nuc_info = constants.nucleus_constants(first_nifti.nucleus[0])
            if nuc_info.ppm_range:
                ppmlim = nuc_info.ppm_range
            else:
                ppmlim = (0.2, 4.2)
        else:
            ppmlim = (0.2, 4.2)
    
    # Handle batch indexing
    if batch_index is not None:
        # Plot single batch element
        if batch_index < 0 or batch_index >= len(nifti_list):
            raise IndexError(f"batch_index {batch_index} out of range [0, {len(nifti_list)})")
        
        nifti_to_plot = nifti_list[batch_index]
        
        if FSL_MRS_AVAILABLE and hasattr(nifti_to_plot, 'plot'):
            # Use native NIfTI-MRS plotting
            fig = nifti_to_plot.plot(
                display_dim=display_dim,
                ppmlim=ppmlim,
                plot_avg=plot_avg,
                legend=legend
            )
            if title:
                fig.suptitle(title, fontsize=14, fontweight='bold')
            elif not title and len(nifti_list) > 1:
                fig.suptitle(f"Batch Element {batch_index+1}/{len(nifti_list)}", fontsize=12)
            return fig
        else:
            # Fallback plotting
            return _plot_single_spectrum(
                nifti_to_plot,
                ppmlim=ppmlim,
                title=title or f"Batch Element {batch_index+1}/{len(nifti_list)}"
            )
    
    else:
        # Plot multiple batch elements
        n_to_plot = min(len(nifti_list), max_batch_display)
        
        if n_to_plot == 1:
            # Just plot the single element
            return vis_nifti_mrs_plus(
                data,
                display_dim=display_dim,
                ppmlim=ppmlim,
                plot_avg=plot_avg,
                batch_index=0,
                legend=legend,
                title=title
            )
        
        # Determine grid layout
        if grid_layout is None:
            if n_to_plot <= 2:
                rows, cols = 1, n_to_plot
            elif n_to_plot <= 4:
                rows, cols = 2, 2
            elif n_to_plot <= 6:
                rows, cols = 2, 3
            elif n_to_plot <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, 3
        else:
            rows, cols = grid_layout
        
        # Create figure
        if figsize is None:
            figsize = (5 * cols, 3.5 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Plot each batch element
        for i in range(n_to_plot):
            nifti = nifti_list[i]
            ax = axes[i]
            
            # Plot spectrum on this axis
            _plot_spectrum_on_axis(
                nifti,
                ax=ax,
                ppmlim=ppmlim,
                title=f"Batch {i+1}",
                legend=False
            )
        
        # Hide unused subplots
        for i in range(n_to_plot, len(axes)):
            axes[i].axis('off')
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        else:
            fig.suptitle(
                f"NIfTI_MRS_Plus Batch ({n_to_plot}/{len(nifti_list)} displayed)",
                fontsize=14,
                fontweight='bold',
                y=0.995
            )
        
        plt.tight_layout()
        return fig


def _plot_single_spectrum(nifti, ppmlim=(0.2, 4.2), title="MRS Spectrum"):
    """Plot a single NIfTI-MRS spectrum using basic matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    _plot_spectrum_on_axis(nifti, ax, ppmlim, title)
    plt.tight_layout()
    return fig


def _plot_spectrum_on_axis(nifti, ax, ppmlim=(0.2, 4.2), title="", legend=True):
    """Plot spectrum on axis, properly handling raw multi-coil/average data."""
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.core.mrs import MRS
    
    # Process data like vis_nifti_mrs does
    processed = nifti
    
    # Handle coil combination
    dim_tags = getattr(processed, 'dim_tags', None)
    # Check if dim_tags is iterable (not None, not Mock)
    try:
        has_coil_dim = dim_tags is not None and 'DIM_COIL' in dim_tags
    except TypeError:
        has_coil_dim = False
    
    if has_coil_dim and processed.shape[processed.dim_position('DIM_COIL')] > 1:
        processed = nifti_mrs_proc.coilcombine(processed)
    
    # Handle all other dimensions
    if dim_tags:
        try:
            for dim in dim_tags:
                if dim is None:
                    continue
                if processed.shape[processed.dim_position(dim)] > 1:
                    if dim in ('DIM_EDIT', 'DIM_METCYCLE', 'DIM_ISIS'):
                        processed = nifti_mrs_proc.subtract(processed, dim=dim)
                    else:
                        processed = nifti_mrs_proc.average(processed, dim)
        except TypeError:
            # dim_tags is not iterable (e.g., Mock object)
            pass
    
    # Create MRS object
    mrs = MRS(
        processed[:].squeeze(),
        bw=processed.bandwidth,
        cf=processed.spectrometer_frequency[0],
        nucleus=processed.nucleus[0]
    )
    
    # Get spectrum
    ppm = mrs.getAxes(ppmlim=ppmlim)
    spec = mrs.get_spec(ppmlim=ppmlim)
    
    # Plot
    ax.plot(ppm, np.real(spec), 'k-', linewidth=1.0, label='Real' if legend else None)
    
    # Formatting
    ax.invert_xaxis()
    if ppmlim is not None:
        ax.set_xlim(ppmlim[1], ppmlim[0])
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=10)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    if legend:
        ax.legend(loc='best', fontsize=8)


def plot_batch_comparison(
    nifti_plus,
    indices=None,
    ppmlim=(0.2, 4.2),
    labels=None,
    title="Batch Comparison",
    colors=None,
    alpha=0.7,
    figsize=(12, 5)
):
    """
    Plot multiple spectra from a batch overlaid for comparison.
    
    Args:
        nifti_plus: NIfTI_MRS_Plus object
        indices: List of batch indices to plot (None = all up to 6)
        ppmlim: PPM limits (min, max)
        labels: List of labels for each spectrum
        title: Plot title
        colors: List of colors for each spectrum
        alpha: Transparency for overlaid spectra
        figsize: Figure size
        
    Returns:
        matplotlib figure
        
    Example:
        >>> plot_batch_comparison(nifti_plus, indices=[0, 1, 2])
        >>> plot_batch_comparison(nifti_plus, labels=['Original', 'Augmented 1', 'Augmented 2'])
    """
    from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus
    
    if not isinstance(nifti_plus, NIfTI_MRS_Plus):
        raise TypeError("nifti_plus must be a NIfTI_MRS_Plus object")
    
    nifti_list = nifti_plus.list()
    
    if indices is None:
        indices = list(range(min(6, len(nifti_list))))
    
    if labels is None:
        labels = [f"Spectrum {i+1}" for i in indices]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))
    
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.core.mrs import MRS
    from fsl_mrs.utils.plotting import plot_spectra
    
    # Process each spectrum properly
    mrs_list = []
    for idx in indices:
        if idx >= len(nifti_list):
            warnings.warn(f"Index {idx} out of range, skipping")
            continue
        
        nifti = nifti_list[idx]
        
        # Process like vis_nifti_mrs
        processed = nifti
        dim_tags = getattr(processed, 'dim_tags', None)
        try:
            has_coil_dim = dim_tags is not None and 'DIM_COIL' in dim_tags
        except TypeError:
            has_coil_dim = False
        
        if has_coil_dim and processed.shape[processed.dim_position('DIM_COIL')] > 1:
            processed = nifti_mrs_proc.coilcombine(processed)
        
        if dim_tags:
            try:
                for dim in dim_tags:
                    if dim is None:
                        continue
                    if processed.shape[processed.dim_position(dim)] > 1:
                        if dim in ('DIM_EDIT', 'DIM_METCYCLE', 'DIM_ISIS'):
                            processed = nifti_mrs_proc.subtract(processed, dim=dim)
                        else:
                            processed = nifti_mrs_proc.average(processed, dim)
            except TypeError:
                pass
        
        # Create MRS object
        mrs = MRS(
            processed[:].squeeze(),
            bw=processed.bandwidth,
            cf=processed.spectrometer_frequency[0],
            nucleus=processed.nucleus[0]
        )
        mrs_list.append(mrs)
    
    # Use FSL-MRS plot_spectra for overlay in ONE figure
    fig = plot_spectra(mrs_list, ppmlim=ppmlim, legend=True)
    
    # Customize
    if fig and len(fig.axes) > 0:
        ax = fig.axes[0]
        ax.set_title(title, fontsize=14, fontweight='bold')
        if len(labels) == len(mrs_list):
            ax.legend(labels, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_batch_grid_detailed(
    nifti_plus,
    max_display=9,
    ppmlim=(0.2, 4.2),
    title="Batch Grid",
    show_metabolites=False,
    figsize=None
):
    """
    Plot batch elements in a detailed grid with optional metabolite regions.
    
    Args:
        nifti_plus: NIfTI_MRS_Plus object
        max_display: Maximum number of spectra to display
        ppmlim: PPM limits
        title: Overall title
        show_metabolites: If True, highlight common metabolite regions
        figsize: Figure size (auto if None)
        
    Returns:
        matplotlib figure
    """
    from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus
    
    if not isinstance(nifti_plus, NIfTI_MRS_Plus):
        raise TypeError("nifti_plus must be a NIfTI_MRS_Plus object")
    
    nifti_list = nifti_plus.list()
    n_to_plot = min(len(nifti_list), max_display)
    
    # Grid layout
    if n_to_plot <= 3:
        rows, cols = 1, n_to_plot
    elif n_to_plot <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    if figsize is None:
        figsize = (5 * cols, 3.5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Metabolite regions (1H MRS)
    metabolite_regions = {
        'NAA': (1.9, 2.1, '#E74C3C'),
        'Cr': (2.9, 3.1, '#3498DB'),
        'Cho': (3.1, 3.3, '#2ECC71'),
    }
    
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.core.mrs import MRS
    
    for i in range(n_to_plot):
        ax = axes[i]
        nifti = nifti_list[i]
        
        # Process like vis_nifti_mrs
        processed = nifti
        dim_tags = getattr(processed, 'dim_tags', None)
        try:
            has_coil_dim = dim_tags is not None and 'DIM_COIL' in dim_tags
        except TypeError:
            has_coil_dim = False
        
        if has_coil_dim and processed.shape[processed.dim_position('DIM_COIL')] > 1:
            processed = nifti_mrs_proc.coilcombine(processed)
        
        if dim_tags:
            try:
                for dim in dim_tags:
                    if dim is None:
                        continue
                    if processed.shape[processed.dim_position(dim)] > 1:
                        if dim in ('DIM_EDIT', 'DIM_METCYCLE', 'DIM_ISIS'):
                            processed = nifti_mrs_proc.subtract(processed, dim=dim)
                        else:
                            processed = nifti_mrs_proc.average(processed, dim)
            except TypeError:
                pass
        
        # Create MRS object and get spectrum
        mrs = MRS(
            processed[:].squeeze(),
            bw=processed.bandwidth,
            cf=processed.spectrometer_frequency[0],
            nucleus=processed.nucleus[0]
        )
        
        ppm = mrs.getAxes(ppmlim=ppmlim)
        spec = mrs.get_spec(ppmlim=ppmlim)
        
        # Plot
        ax.plot(ppm, np.real(spec), 'k-', linewidth=1.0)
        
        # Highlight metabolites
        if show_metabolites:
            for name, (ppm_min, ppm_max, color) in metabolite_regions.items():
                ax.axvspan(ppm_min, ppm_max, alpha=0.15, color=color)
        
        ax.invert_xaxis()
        if ppmlim:
            ax.set_xlim(ppmlim[1], ppmlim[0])
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_title(f"Sample {i+1}", fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Only show x-label on bottom row
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Chemical Shift (ppm)", fontsize=9)
    
    # Hide unused subplots
    for i in range(n_to_plot, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    return fig


# Convenience function for interactive use
def quick_plot(nifti_plus, index=0, ppmlim=(0.2, 4.2)):
    """
    Quick plot of a single spectrum from batch.
    
    Args:
        nifti_plus: NIfTI_MRS_Plus object
        index: Batch index to plot
        ppmlim: PPM range
        
    Example:
        >>> quick_plot(nifti_plus, 0)  # Plot first spectrum
    """
    return vis_nifti_mrs_plus(nifti_plus, batch_index=index, ppmlim=ppmlim)



