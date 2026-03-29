####################################################################################################
#                                        zero_fill.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-03-29                                                                              #
#                                                                                                  #
# Purpose: Zero-fill (zero-pad) or crop FID to a fixed target number of spectral points.           #
#                                                                                                  #
####################################################################################################

import numpy as np
from typing import Optional, List

from augmentrum.core.base_module import BaseModule
from augmentrum.utils.tensor_ops import _is_torch, _is_jax, _is_tf
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs


class ZeroFill(BaseModule):
    """
    Zero-pad (or crop) the FID to a fixed target number of spectral points.

    Zero-filling increases apparent spectral resolution by interpolating
    between frequency bins.  Cropping (``target_pts < N_PTS``) is equivalent
    to ``Apodization(mode='truncate')``.

    Because ``target_pts`` is a module-level scalar, the same length is
    applied to **every item in the batch** — so the output batch remains
    uniform regardless of backend.

    On non-NIfTI tensor backends, if N_PTS changes the base class detects
    the shape change, rebuilds each NIfTI_MRS object at the new length, and
    emits a ``RuntimeWarning`` (the "medium-speed" path).  For maximum
    throughput keep N_PTS the same throughout the whole pipeline.

    Parameters
    ----------
    target_pts : int
        Target number of spectral points after zero-filling / cropping.

    Examples
    --------
    >>> zf = ZeroFill(target_pts=2048)   # double a 1024-point FID
    >>> out, _ = zf(nifti_plus, None)
    """

    SUPPORTED_BACKENDS = []  # All backends via process_tensor

    def __init__(self, target_pts: int):
        super().__init__(target_pts=target_pts)
        if int(target_pts) < 1:
            raise ValueError(f"target_pts must be >= 1, got {target_pts}")
        self.target_pts = int(target_pts)

    # ── NIfTI-list path ────────────────────────────────────────────────────────

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """Apply zero-fill to each NIFTI_MRS object in the list."""
        processed_data = []
        for nifti in data_list:
            fid = nifti[:]
            fid_out = self._apply_numpy(fid)

            if fid_out.shape[-1] != fid.shape[-1]:
                # N_PTS changed — rebuild NIfTI_MRS with new length
                try:
                    affine = nifti.getAffine('voxel', 'world')
                except Exception:
                    affine = None
                nucleus = (nifti.nucleus[0]
                           if hasattr(nifti, 'nucleus') and nifti.nucleus else '1H')
                dim_tags = (list(nifti.dim_tags)
                            if hasattr(nifti, 'dim_tags') else [None, None, None])
                new_nifti = gen_nifti_mrs(
                    data=fid_out,
                    dwelltime=nifti.dwelltime,
                    spec_freq=nifti.spectrometer_frequency[0],
                    nucleus=nucleus,
                    dim_tags=dim_tags,
                    affine=affine,
                )
                if hasattr(nifti, 'hdr_ext'):
                    for key in nifti.hdr_ext:
                        if key not in ('SpectrometerFrequency', 'ResonantNucleus',
                                       'dim_5', 'dim_6', 'dim_7'):
                            try:
                                new_nifti.add_hdr_field(key, nifti.hdr_ext[key])
                            except Exception:
                                pass
                processed_data.append(new_nifti)
            else:
                nifti[:] = fid_out
                processed_data.append(nifti)

        return processed_data, water_list

    # ── Tensor path (all non-NIfTI backends) ──────────────────────────────────

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """Apply zero-fill natively in whatever backend format the data is in.

        Cropping (target_pts < N_PTS) is a slice — free.
        Padding (target_pts > N_PTS) creates a zeros tensor in the native
        backend and concatenates along the last axis — no numpy round-trip,
        gradients preserved.
        """
        n_in  = data_array.shape[-1]
        n_out = self.target_pts

        if n_out == n_in:
            return data_array, water_array

        if n_out < n_in:
            # Crop — native slice, works in every backend
            return data_array[..., :n_out], water_array

        # ── Zero-pad ──────────────────────────────────────────────────────────
        pad_n     = n_out - n_in
        pad_shape = list(data_array.shape)
        pad_shape[-1] = pad_n

        if _is_torch(data_array):
            import torch
            zeros = torch.zeros(pad_shape, dtype=data_array.dtype, device=data_array.device)
            return torch.cat([data_array, zeros], dim=-1), water_array

        if _is_jax(data_array):
            import jax.numpy as jnp
            zeros = jnp.zeros(pad_shape, dtype=data_array.dtype)
            return jnp.concatenate([data_array, zeros], axis=-1), water_array

        if _is_tf(data_array):
            import tensorflow as tf
            zeros = tf.zeros(pad_shape, dtype=data_array.dtype)
            return tf.concat([data_array, zeros], axis=-1), water_array

        # NumPy (and Keras when backed by NumPy / TF caught above)
        dt    = data_array.dtype if hasattr(data_array, 'dtype') else np.complex64
        zeros = np.zeros(pad_shape, dtype=dt)
        try:
            return np.concatenate([data_array, zeros], axis=-1), water_array
        except TypeError:
            # Keras tensor — fall back to TF ops
            import tensorflow as tf
            zeros_tf = tf.zeros(pad_shape, dtype=data_array.dtype)
            return tf.concat([data_array, zeros_tf], axis=-1), water_array

    # ── Internal numpy helper (used by process_nifti_list) ───────────────────

    def _apply_numpy(self, fid: np.ndarray) -> np.ndarray:
        n_in  = fid.shape[-1]
        n_out = self.target_pts
        if n_out == n_in:
            return fid
        if n_out < n_in:
            return fid[..., :n_out]
        pad_width = [(0, 0)] * (fid.ndim - 1) + [(0, n_out - n_in)]
        return np.pad(fid, pad_width)


