####################################################################################################
#                                      gaussian_noise.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: K. C. Igwe (kci2104@columbia.edu)                                                       #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Implements uncorrelated complex Gaussian noise (AWGN) for MRS data augmentation.        #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np
from typing import Optional, List

from augmentrum.core.base_module import BaseModule
from augmentrum.utils.tensor_ops import to_numpy, match_backend, is_torch, is_jax, is_tf


#**************************************************************************************************#
#                                       Class GaussianNoise                                        #
#**************************************************************************************************#
#                                                                                                  #
# Add uncorrelated complex Gaussian noise (AWGN) to MRS data.                                      #
#                                                                                                  #
#**************************************************************************************************#
class GaussianNoise(BaseModule):
    """
    Add uncorrelated complex Gaussian noise (AWGN) to MRS data.

    **Backend-agnostic** (zea pattern): ``process_tensor`` works transparently
    with NumPy, PyTorch, JAX, or TensorFlow tensors.

    Parameters
    ----------
    snr: float, optional
        Signal-to-noise ratio (unitless, signal_power / noise_power)
    snr_db : float, optional
        Signal-to-noise ratio in dB (signal_power / noise_power)
    sigma : float, optional
        Standard deviation per dimension (real and imaginary)
    sigma_frac : float, optional
        Sigma as fraction of max|FID| (alternative to sigma)
    seed : int, optional
        Random seed for reproducibility
    global_scale : bool, default False
        Controls how the reference statistic for ``snr`` / ``snr_db`` /
        ``sigma_frac`` is reduced.  ``False`` (default) measures it **per FID
        trace**, which is the right thing for SVS.  ``True`` measures a single
        statistic **per batch element**, reduced over every other axis.

        Use ``global_scale=True`` for MRSI.  A per-trace statistic on a
        ``(B, X, Y, Z, T)`` volume gives every voxel its own sigma, so
        background voxels (whose signal power is ~0) receive almost no noise
        while brain voxels receive plenty.  The resulting noise field traces the
        anatomy and hands a network a free brain mask.  ``sigma`` is an absolute
        scale and is unaffected by this flag.

    Notes
    -----
    Provide EITHER snr_db, sigma, OR sigma_frac (not multiple).
    If snr_db is used, sigma is computed from mean(|fid|^2).

    Examples
    --------
    >>> noise = GaussianNoise(snr=20.0)
    >>> noise = GaussianNoise(snr_db=10.0)
    >>> noise = GaussianNoise(sigma=0.01)
    >>> noise = GaussianNoise(sigma_frac=0.02)
    >>> noise = GaussianNoise(sigma_frac=0.02, global_scale=True)  # MRSI volumes
    """

    SUPPORTED_BACKENDS = []  # Supports all backends

    def __init__(self,
                 snr: Optional[float] = None,
                 snr_db: Optional[float] = None,
                 sigma: Optional[float] = None,
                 sigma_frac: Optional[float] = None,
                 seed: Optional[int] = None,
                 global_scale: bool = False):
        super().__init__(snr=snr, snr_db=snr_db, sigma=sigma, sigma_frac=sigma_frac,
                         seed=seed, global_scale=global_scale)

        self.snr = snr
        self.snr_db = snr_db
        self.sigma = sigma
        self.sigma_frac = sigma_frac
        self.seed = seed
        self.global_scale = global_scale

        # Validate parameters
        params_provided = sum([snr is not None, snr_db is not None,
                               sigma is not None, sigma_frac is not None])
        if params_provided == 0:
            raise ValueError("Must provide one of: snr, snr_db, sigma, or sigma_frac")
        if params_provided > 1:
            raise ValueError("Provide only ONE of: snr, snr_db, sigma, or sigma_frac")

    def process_nifti_list(self, data_list: List, water_list: Optional[List] = None, **kwargs):
        """Add Gaussian noise to list of NIFTI_MRS objects."""
        processed_data = []
        for nifti in data_list:
            fid = nifti[:]
            fid_noisy = self._add_noise_numpy(fid)
            nifti[:] = fid_noisy
            processed_data.append(nifti)
        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add Gaussian noise to tensor/array data (**any backend, natively**).

        Scale statistics (sigma_frac / snr modes) are measured with a tiny
        ``to_numpy`` scalar reduction.  Noise is then **generated in the
        native framework's own RNG** — ``torch.randn`` for PyTorch,
        ``jax.random.normal`` for JAX, ``tf.random.normal`` for TF/Keras,
        or ``numpy`` as the universal fallback.  This keeps gradients intact
        and avoids the old "numpy noise → match_backend" workaround that
        broke under JIT tracing.

        Args:
            data_array: Input tensor of shape ``(batch, ..., n_points)``
            water_array: Optional water reference tensor (unchanged)
            backend: Backend enum (unused — ops dispatch automatically)

        Returns:
            Tuple of (noisy_data, water_array)
        """
        original_shape = tuple(data_array.shape)
        ndim = len(original_shape)

        # ── Step 1: compute noise scale (small scalar reduction → numpy OK) ──
        # `keepdims=True` throughout, so scale_np always broadcasts against
        # original_shape without any further reshaping.  Reducing to a flat
        # (n_batch, 1) and reshaping only broadcasts correctly when every axis
        # between the batch and the points axis is singleton — true for SVS
        # (B, 1, 1, 1, N), false for MRSI (B, X, Y, Z, T).
        if self.sigma is not None:
            # Fixed sigma — no data stats needed at all
            scale_np = np.full((1,) * ndim, float(self.sigma), dtype=np.float64)
        else:
            arr_np = to_numpy(data_array)
            # global_scale: one statistic per batch element; otherwise per trace.
            red_axes = tuple(range(1, ndim)) if self.global_scale else (ndim - 1,)

            if self.sigma_frac is not None:
                peak = np.max(np.abs(arr_np), axis=red_axes, keepdims=True)
                peak = np.where(peak > 0, peak, 1.0)
                scale_np = self.sigma_frac * peak
            else:
                sig_pow = np.mean(np.abs(arr_np) ** 2, axis=red_axes, keepdims=True) + 1e-16
                if self.snr is not None:
                    noise_pow = sig_pow / float(self.snr)
                else:
                    noise_pow = sig_pow / (10.0 ** (float(self.snr_db) / 10.0))
                scale_np = np.sqrt(noise_pow / 2.0)  # per-channel std

        # ── Step 2: generate noise in the native framework ────────────────────
        if is_torch(data_array):
            import torch
            gen = torch.Generator(device=data_array.device)
            if self.seed is not None:
                gen.manual_seed(self.seed)
            # Determine real dtype for noise
            if data_array.is_complex():
                rdtype = data_array.real.dtype
            else:
                rdtype = data_array.dtype
            scale_t = torch.as_tensor(scale_np, device=data_array.device, dtype=rdtype)
            r = torch.randn(*original_shape, generator=gen, device=data_array.device, dtype=rdtype)
            i_ = torch.randn(*original_shape, generator=gen, device=data_array.device, dtype=rdtype)
            noise = torch.complex(r, i_) * scale_t
            return data_array + noise, water_array

        elif is_jax(data_array):
            import jax
            import jax.numpy as jnp
            seed = self.seed if self.seed is not None else 0
            key = jax.random.PRNGKey(seed)
            k1, k2 = jax.random.split(key)
            r = jax.random.normal(k1, shape=original_shape, dtype=jnp.float32)
            i_ = jax.random.normal(k2, shape=original_shape, dtype=jnp.float32)
            scale_j = jnp.array(scale_np, dtype=jnp.float32)
            noise = (r + 1j * i_).astype(jnp.complex64) * scale_j
            return data_array + noise, water_array

        elif is_tf(data_array):
            import tensorflow as tf
            if self.seed is not None:
                tf.random.set_seed(self.seed)
            r = tf.random.normal(original_shape, dtype=tf.float32)
            i_ = tf.random.normal(original_shape, dtype=tf.float32)
            scale_t = tf.constant(scale_np, dtype=tf.float32)
            noise = tf.complex(r * scale_t, i_ * scale_t)
            return data_array + noise, water_array

        else:
            # NumPy and any other framework (e.g. Keras-native): numpy noise + match_backend
            rng = np.random.default_rng(self.seed)
            r = rng.normal(0.0, 1.0, size=original_shape)
            i_ = rng.normal(0.0, 1.0, size=original_shape)
            noise = (r + 1j * i_) * scale_np
            return data_array + match_backend(noise, data_array), water_array

    #********************************************************************#
    #   pure-numpy path for process_nifti_list (input is always numpy)   #
    #********************************************************************#

    def _add_noise_numpy(self, fid: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to numpy FID data."""
        rng = np.random.default_rng(self.seed)

        original_shape = fid.shape
        fid_flat = fid.reshape(-1, original_shape[-1])
        out_dtype = np.result_type(fid.dtype, np.complex64)
        fid_flat = fid_flat.astype(out_dtype, copy=False)

        # This path handles one NIFTI_MRS object at a time, so "global" means a
        # single statistic over the whole array rather than one per FID trace.
        red_axis = None if self.global_scale else -1

        if self.sigma is not None:
            scale = float(self.sigma)
        elif self.sigma_frac is not None:
            peak = np.max(np.abs(fid_flat), axis=red_axis, keepdims=True)
            peak = np.where(peak > 0, peak, 1.0)
            scale = self.sigma_frac * peak
        else:
            sig_pow = np.mean(np.abs(fid_flat) ** 2, axis=red_axis, keepdims=True) + 1e-16
            if self.snr is not None:
                noise_pow = sig_pow / float(self.snr)
            else:
                noise_pow = sig_pow / (10.0 ** (float(self.snr_db) / 10.0))
            scale = np.sqrt(noise_pow / 2.0)

        noise = (rng.normal(0.0, 1.0, size=fid_flat.shape)
                 + 1j * rng.normal(0.0, 1.0, size=fid_flat.shape)) * scale

        return (fid_flat + noise).reshape(original_shape)
