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

#*************#
#   imports   #
#*************#
import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
import warnings


#*********************************#
#   simple spectrum / fit plots   #
#*********************************#
def _crop_ppm(arr, ppmAxis, ppmLim):
    """Return (arr_cropped, ppmAxis_cropped) within *ppmLim* = (lo, hi)."""
    beg = int((np.abs(ppmAxis - ppmLim[0])).argmin())
    end = int((np.abs(ppmAxis - ppmLim[1])).argmin())
    if beg > end:
        beg, end = end, beg
    return arr[beg:end], ppmAxis[beg:end]


def _pick_part(arr, real):
    """Select real or imaginary part of a (possibly already real) array."""
    return np.real(arr) if real else np.imag(arr)


def _style_mrs_ax(ax, clean):
    """Canonical MRS axis style (inverted ppm, y-hidden); everything hidden when *clean*."""
    for spine in ('right', 'top', 'left'):
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.invert_xaxis()
    if clean:
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticks_position('none')


def _save_fig(fig, save_path, name):
    """Save *fig* to "save_path/name.png", close it, and return the path."""
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f'{name}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)
        return out
    return None


def plot_spec_and_fit(spec, fit, true=None, ppmAxis=None, ppmLim=None, name='',
                      save_path=None, real=True, clean=False):
    """Plot a measured spectrum, the model fit, and the residual on a ppm axis.

    Visual style: Data (black) / Fit (red, α=0.6) / Residual offset above (dimgray) /
    optional True spectrum (green dashed).  ppm-axis inverted, y-axis hidden.

    Parameters
    ----------
    spec      : np.ndarray (complex)  measured spectrum (1-D).
    fit       : np.ndarray (complex)  forward-modeled fit (1-D).
    true      : np.ndarray (complex), optional  ground-truth spectrum.
    ppmAxis   : np.ndarray, optional  chemical-shift axis matching *spec*. Defaults to
                "linspace(0.5, 4.0, len(spec))".
    ppmLim    : tuple(float, float), optional  (lo, hi) ppm window to display.
    name      : str  filename stem (without extension).
    save_path : str, optional  directory to save PNG into; created if missing.
    real      : bool  if True plot real part, else imaginary.
    clean     : bool  if True hide all axes/labels (publication snippet style).
    """
    if ppmAxis is None:
        ppmAxis = np.linspace(0.5, 4.0, spec.shape[0])

    if ppmLim is not None:
        orig_ppm = ppmAxis                                    # save BEFORE first crop
        spec, ppmAxis = _crop_ppm(spec, orig_ppm, ppmLim)
        fit_ppm = orig_ppm if fit.shape[0] == orig_ppm.shape[0] \
                  else np.linspace(orig_ppm[0], orig_ppm[-1], fit.shape[0])
        fit, _ = _crop_ppm(fit, fit_ppm, ppmLim)
        if true is not None:
            true_ppm = orig_ppm if true.shape[0] == orig_ppm.shape[0] \
                       else np.linspace(orig_ppm[0], orig_ppm[-1], true.shape[0])
            true, _ = _crop_ppm(true, true_ppm, ppmLim)

    spec = _pick_part(spec, real)
    fit  = _pick_part(fit,  real)
    residual = spec - fit + 1.1 * spec.max()

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(ppmAxis, spec,     'k',       label='Data',     linewidth=1)
    ax.plot(ppmAxis, fit,      'r',       label='Fit',      alpha=0.6, linewidth=2)
    ax.plot(ppmAxis, residual, 'dimgray', label='Residual', alpha=0.8, linewidth=1)
    if true is not None:
        true = _pick_part(true, real)
        ax.plot(ppmAxis, true, 'g', label='True Spectrum',
                alpha=0.8, linewidth=1, linestyle='--')

    if not clean:
        ax.set_xlabel('Chemical Shift [ppm]')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style_mrs_ax(ax, clean)
    return _save_fig(fig, save_path, name)


def plot_spec(spec, ppmAxis=None, ppmLim=None, name='', save_path=None,
              real=True, color='k', title=None, clean=False):
    """Plot a single complex spectrum on a ppm axis (no fit / residual).

    Same visual style as "plot_spec_and_fit" — useful for sanity-checking
    the simulated input in isolation.

    Parameters
    ----------
    spec      : np.ndarray (complex)  spectrum to display (1-D).
    ppmAxis   : np.ndarray, optional  chemical-shift axis matching *spec*.
    ppmLim    : tuple(float, float), optional  (lo, hi) ppm window to display.
    name      : str  filename stem (without extension).
    save_path : str, optional  directory to save PNG into; created if missing.
    real      : bool  if True plot real part, else imaginary.
    color     : str  matplotlib color string for the spectrum line.
    title     : str, optional  axes title (small font, publication-friendly).
    clean     : bool  if True hide all axes/labels (publication snippet style).
    """
    if ppmAxis is None:
        ppmAxis = np.linspace(0.5, 4.0, spec.shape[0])

    if ppmLim is not None:
        spec, ppmAxis = _crop_ppm(spec, ppmAxis, ppmLim)

    spec = _pick_part(spec, real)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(ppmAxis, spec, color, label='Data', linewidth=1)
    if not clean:
        ax.set_xlabel('Chemical Shift [ppm]')
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style_mrs_ax(ax, clean)
    return _save_fig(fig, save_path, name)


#******************************#
#   nifti-mrs batch plotting   #
#******************************#
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
    from nifti_mrs_plus import NIfTI_MRS_Plus
    
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
    from nifti_mrs_plus import NIfTI_MRS_Plus
    
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
    from nifti_mrs_plus import NIfTI_MRS_Plus
    
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





#****************#
#   mrsi plots   #
#****************#
#
#  Color choices follow the jobs the data does:
#    * a metabolite / intensity map encodes MAGNITUDE  -> one perceptually
#      uniform sequential ramp (viridis), never a rainbow
#    * a sampling mask encodes PRESENCE                -> binary grayscale
#    * overlaid spectra encode IDENTITY                -> a fixed categorical
#      order, assigned by position and never cycled, so "clean" is the same
#      color in every figure of a run
#
#  _MRSI_SERIES is deliberately short. Past four overlaid traces a spectrum plot
#  stops being readable and wants small multiples instead.

_MRSI_SERIES = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd')
_MRSI_SEQUENTIAL = 'viridis'


def _band_integral(volume, ppm_axis, band, axis=-1):
    """
    Collapse the spectral axis to one number per voxel by integrating |spectrum|
    over a ppm band. This is what makes a 4-D MRSI volume mappable.
    """
    lo, hi = (min(band), max(band))
    sel = (ppm_axis >= lo) & (ppm_axis <= hi)
    if not sel.any():
        raise ValueError(
            f"ppm band {band} selects no points from an axis spanning "
            f"{ppm_axis.min():.2f}..{ppm_axis.max():.2f} ppm."
        )
    return np.abs(np.take(volume, np.flatnonzero(sel), axis=axis)).sum(axis=axis)


def plot_mrsi_map(volume, ppm_axis=None, band=(1.8, 2.2), slices=None,
                  n_cols=None, title=None, cmap=_MRSI_SEQUENTIAL, mask=None,
                  vmax_percentile=99.5, save_path=None, name='mrsi_map',
                  figsize=None, cbar_label=None):
    """
    Montage of axial slices through an MRSI volume.

    Args:
        volume: (X, Y, Z, T) complex spectral volume, or (X, Y, Z) real map.
                A 4-D input is reduced by integrating |spectrum| over *band*.
        ppm_axis: chemical-shift axis matching the last axis. Required for 4-D.
        band: ppm window to integrate. Default (1.8, 2.2) brackets NAA at 2.008.
        slices: which z indices to show. Default is up to 9 spread through the slab.
        mask: optional (X, Y, Z) boolean; voxels outside are blanked.
        vmax_percentile: upper display limit, as a percentile of the shown voxels.
                Guards against one hot voxel flattening everything else.
        cbar_label: color-bar label. Defaults to the integrated band.

    Returns:
        The saved path when *save_path* is given, else the Figure.
    """
    volume = np.asarray(volume)
    if volume.ndim == 4:
        if ppm_axis is None:
            raise ValueError("ppm_axis is required to reduce a 4-D volume to a map.")
        img = _band_integral(volume, np.asarray(ppm_axis), band)
        default_label = f'|signal| integrated over {min(band)}-{max(band)} ppm'
    elif volume.ndim == 3:
        img = np.abs(volume)
        default_label = 'amplitude'
    else:
        raise ValueError(f"volume must be 3-D or 4-D, got shape {volume.shape}.")

    if mask is not None:
        img = np.where(np.asarray(mask).astype(bool), img, np.nan)

    n_z = img.shape[2]
    if slices is None:
        slices = list(np.unique(np.linspace(0, n_z - 1, min(9, n_z)).astype(int)))
    slices = [int(s) for s in slices]

    finite = img[np.isfinite(img)]
    vmax = np.percentile(finite, vmax_percentile) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    n_cols = n_cols or min(len(slices), 3)
    n_rows = int(np.ceil(len(slices) / n_cols))
    figsize = figsize or (3.2 * n_cols, 3.4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    im = None
    for k, ax in enumerate(axes.ravel()):
        if k >= len(slices):
            ax.axis('off')
            continue
        z = slices[k]
        im = ax.imshow(np.rot90(img[:, :, z]), cmap=cmap, vmin=0, vmax=vmax,
                       interpolation='nearest')
        # The slice index is the only per-panel label; a full axis frame on a
        # montage is noise.
        ax.set_title(f'z = {z}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(cbar_label or default_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    saved = _save_fig(fig, save_path, name)
    return saved if saved is not None else fig


def plot_mrsi_voxel_spectra(volumes, ppm_axis, voxel, labels=None,
                            ppm_lim=(0.2, 4.2), real=True, title=None,
                            normalize=False, save_path=None,
                            name='mrsi_voxel_spectra', figsize=(11, 4.5)):
    """
    Overlay the spectrum of one voxel across several versions of a volume.

    The intended use is showing what an augmentation did: pass the clean volume
    and its undersampled / noisy counterparts and read off the difference at a
    single voxel.

    Args:
        volumes: one spectral volume, or a sequence of them. Each is (X, Y, Z, T)
                complex, already transformed to spectra.
        ppm_axis: chemical-shift axis matching the last axis.
        voxel: (x, y, z) index to plot.
        labels: one per volume. Needed once there is more than one.
        ppm_lim: display window. Default (0.2, 4.2) is the usual metabolite range.
        real: plot the real part (the convention for a phased spectrum) or the
                magnitude when False.
        normalize: scale each trace to unit maximum. Use when comparing shape
                across volumes whose absolute scales differ.

    Returns:
        The saved path when *save_path* is given, else the Figure.
    """
    if not isinstance(volumes, (list, tuple)):
        volumes = [volumes]
    if labels is None:
        labels = [f'volume {i}' for i in range(len(volumes))] if len(volumes) > 1 else [None]
    if len(labels) != len(volumes):
        raise ValueError(f"got {len(volumes)} volumes but {len(labels)} labels.")
    if len(volumes) > len(_MRSI_SERIES):
        raise ValueError(
            f"{len(volumes)} overlaid spectra is past readable; "
            f"plot at most {len(_MRSI_SERIES)}, or use separate panels."
        )

    ppm_axis = np.asarray(ppm_axis)
    vx, vy, vz = (int(v) for v in voxel)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, (vol, label) in enumerate(zip(volumes, labels)):
        trace = np.asarray(vol)[vx, vy, vz]
        y = np.real(trace) if real else np.abs(trace)
        y, x = _crop_ppm(y, ppm_axis, ppm_lim)
        if normalize:
            peak = np.max(np.abs(y))
            y = y / peak if peak > 0 else y
        ax.plot(x, y, color=_MRSI_SERIES[i], linewidth=1.4, label=label)

    ax.set_xlabel('chemical shift [ppm]', fontsize=11)
    _style_mrs_ax(ax, clean=False)
    ax.set_xlim(max(ppm_lim), min(ppm_lim))
    if any(l is not None for l in labels):
        ax.legend(frameon=False, fontsize=10)
    ax.set_title(title or f'voxel ({vx}, {vy}, {vz})', fontsize=12)
    fig.tight_layout()

    saved = _save_fig(fig, save_path, name)
    return saved if saved is not None else fig


def plot_kspace_mask(mask, title=None, save_path=None, name='kspace_mask',
                     figsize=None, subtitles=None):
    """
    Show one or more k-space sampling masks, with the retained fraction stated.

    Args:
        mask: (kx, ky) boolean, or a sequence / stack of them.
        subtitles: per-panel labels. Defaults to the achieved acceleration, which
                is the number you actually want to check.

    Returns:
        The saved path when *save_path* is given, else the Figure.
    """
    mask = np.asarray(mask)
    if mask.ndim == 2:
        masks = [mask]
    elif mask.ndim == 3:
        masks = [mask[i] for i in range(mask.shape[0])]
    else:
        raise ValueError(f"mask must be 2-D or a stack of 2-D, got {mask.shape}.")

    n = len(masks)
    figsize = figsize or (3.1 * n, 3.4)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    for i, (m, ax) in enumerate(zip(masks, axes.ravel())):
        m = np.asarray(m).astype(float)
        while m.ndim > 2:                       # collapse any fully-sampled axes
            m = m[..., 0]
        ax.imshow(m.T, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        kept = float(m.mean())
        ax.set_title(subtitles[i] if subtitles else
                     f'{100 * kept:.0f}% of bins  (1/{1 / kept:.1f})' if kept > 0 else 'empty',
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()

    saved = _save_fig(fig, save_path, name)
    return saved if saved is not None else fig
