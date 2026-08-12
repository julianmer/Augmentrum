####################################################################################################
#                                             noise.py                                             #
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

from abc import ABC, abstractmethod

from augmentrum.core.base_module import BaseModule
from nifti_mrs_plus import Backend, ops


__all__ = ['Noise',
           'NoiseCovariance', 'Independent', 'FromSensitivity', 'SuppliedCovariance',
           'NoiseProfile', 'Flat', 'SuppliedProfile', 'FromNoiseScan']


#**************************************************************************************************#
#                                      Class NoiseCovariance                                       #
#**************************************************************************************************#
#                                                                                                  #
# How the channels of a receive array share their noise.                                           #
#                                                                                                  #
#**************************************************************************************************#
class NoiseCovariance(ABC):
    """
    How the channels of a receive array share their noise.

    Elements of a real array do not see independent noise: they couple through
    mutual inductance and through the sample they all look at. Independence is
    the first-order model and a defensible default, but it makes an array look
    better than it is - correlated channels carry less information than the
    same number of independent ones.

    Stating it as a contract keeps the question of where psi comes from separate
    from the drawing itself, the same way sensitivity maps are handled.
    """

    @abstractmethod
    def matrix(self, n_coils: int) -> np.ndarray:
        """
        Covariance between channels, "(C, C)" Hermitian positive semi-definite.

        Args:
            n_coils: Channels in the array.

        Returns:
            The covariance, with unit diagonal - the overall level is set by the
            noise scale, not here.
        """

    @staticmethod
    def _normalized(psi: np.ndarray) -> np.ndarray:
        """Unit diagonal, so psi says only how the channels relate."""
        scale = np.sqrt(np.abs(np.diag(psi)))
        scale = np.where(scale > 0, scale, 1.0)
        return (psi / scale[:, None] / scale[None, :]).astype(np.complex64)


#**************************************************************************************************#
#                                       Class Independent                                          #
#**************************************************************************************************#
#                                                                                                  #
# Channels that share nothing, which is the usual first-order model.                               #
#                                                                                                  #
#**************************************************************************************************#
class Independent(NoiseCovariance):
    """Channels that share nothing, which is the usual first-order model."""

    def matrix(self, n_coils: int) -> np.ndarray:
        """The identity: every channel draws on its own."""
        return np.eye(n_coils, dtype=np.complex64)


#**************************************************************************************************#
#                                      Class FromSensitivity                                       #
#**************************************************************************************************#
#                                                                                                  #
# Coupling modelled from how much the elements' sensitivities overlap.                             #
#                                                                                                  #
#**************************************************************************************************#
class FromSensitivity(NoiseCovariance):
    """
    Coupling modelled from how much the elements' sensitivities overlap.

    Two elements that see the same part of the sample also share the noise that
    part contributes, so the overlap of their sensitivities stands in for how
    correlated they are.

    This is a **model, not a measurement**, and the distinction matters for what
    can be concluded from it. Real coupling has two sources: the shared sample,
    which the overlap does capture, and mutual inductance between the coils,
    which it cannot - that depends on the geometry and tuning of the array, not
    on what it sees. A study comparing reconstruction methods under realistic
    coupling should measure psi from a noise prescan and pass it to
    :class:"SuppliedCovariance" instead. This is for making an array behave
    plausibly, not for characterising a particular one.

    Args:
        maps: Sensitivity maps "(X, Y, Z, C)", as any MapSource produces.
    """

    def __init__(self, maps):
        self.maps = np.asarray(maps)

    def matrix(self, n_coils: int) -> np.ndarray:
        """The Gram matrix of the maps, Hermitian by construction."""
        if self.maps.shape[-1] != n_coils:
            raise ValueError(
                f"These maps describe {self.maps.shape[-1]} channels but the data "
                f"has {n_coils}. Use the maps the array was built with."
            )
        flat = self.maps.reshape(-1, n_coils)
        return self._normalized(flat.conj().T @ flat)


#**************************************************************************************************#
#                                     Class SuppliedCovariance                                     #
#**************************************************************************************************#
#                                                                                                  #
# A covariance the caller measured.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class SuppliedCovariance(NoiseCovariance):
    """A covariance the caller measured."""

    def __init__(self, psi):
        self.psi = np.asarray(psi)

    def matrix(self, n_coils: int) -> np.ndarray:
        """Hand it over, once it is known to fit."""
        if self.psi.shape != (n_coils, n_coils):
            raise ValueError(
                f"psi is {self.psi.shape} but the data has {n_coils} channels.")
        return self._normalized(self.psi)


#**************************************************************************************************#
#                                       Class NoiseProfile                                         #
#**************************************************************************************************#
#                                                                                                  #
# How loud the noise is, from place to place.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class NoiseProfile(ABC):
    """
    How loud the noise is, from place to place.

    Real noise is not flat across a volume. Receive sensitivity falls off with
    distance from the elements and parallel imaging amplifies it unevenly, so
    the same acquisition is quieter in the middle of the head than at the edge
    of the field of view. Treating it as uniform makes the hard voxels look
    easier than they are.

    A profile is *relative*: it says where the noise is louder, and averages to
    one, so the overall level stays whatever the SNR or sigma asked for.
    """

    @abstractmethod
    def sigma(self, matrix) -> np.ndarray:
        """
        Relative noise level over a spatial grid.

        Args:
            matrix: Grid to cover, "(X, Y, Z)".

        Returns:
            Multipliers "(X, Y, Z)", averaging one.
        """

    @staticmethod
    def _unit_mean(profile: np.ndarray) -> np.ndarray:
        """Scaled to average one, so it says only where, never how much."""
        mean = float(np.mean(profile))
        return (profile / mean if mean > 0 else np.ones_like(profile)).astype(np.float32)


#**************************************************************************************************#
#                                          Class Flat                                              #
#**************************************************************************************************#
#                                                                                                  #
# The same everywhere, which is the usual assumption.                                              #
#                                                                                                  #
#**************************************************************************************************#
class Flat(NoiseProfile):
    """The same everywhere, which is the usual assumption."""

    def sigma(self, matrix) -> np.ndarray:
        """Ones, so nothing is modulated."""
        return np.ones(tuple(int(n) for n in matrix), np.float32)


#**************************************************************************************************#
#                                     Class SuppliedProfile                                        #
#**************************************************************************************************#
#                                                                                                  #
# A profile the caller already has.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class SuppliedProfile(NoiseProfile):
    """A profile the caller already has."""

    def __init__(self, profile):
        self.profile = np.asarray(profile, dtype=np.float32)

    def sigma(self, matrix) -> np.ndarray:
        """Hand it over, once it is known to fit."""
        matrix = tuple(int(n) for n in matrix)
        if self.profile.shape != matrix:
            raise ValueError(
                f"This profile covers {self.profile.shape} but the data is {matrix}.")
        return self._unit_mean(self.profile)


#**************************************************************************************************#
#                                     Class FromNoiseScan                                          #
#**************************************************************************************************#
#                                                                                                  #
# Measured from a noise-only acquisition.                                                          #
#                                                                                                  #
#**************************************************************************************************#
class FromNoiseScan(NoiseProfile):
    """
    Measured from a noise-only acquisition.

    A scan taken with no excitation shows the noise on its own, and how it
    varies across the volume is exactly what is wanted here. Three things about
    such a scan have to be handled rather than assumed:

    - it is a **magnitude**, so its local spread is proportional to sigma but
      not equal to it. That is fine, because only the shape is used;
    - it is usually **masked**, with whole regions set to zero. Those are not
      quiet noise and would drag the estimate down, so they are left out;
    - it is at **imaging resolution**, far finer than any MRSI grid, so it is
      reduced by blocks rather than interpolated - the spread within a block is
      the quantity of interest, and interpolating would smooth it away.

    Args:
        path: A noise-only volume readable by nibabel.
        min_present: What fraction of a block must survive the scan's own mask
            before its spread is believed. A block sitting half in background is
            measuring the edge of the mask rather than the noise there, and its
            spread comes out far too low; those are filled in instead.
    """

    def __init__(self, path, min_present: float = 0.5):
        self.path = str(path)
        self.min_present = float(min_present)
        self._volume = None

    def sigma(self, matrix) -> np.ndarray:
        """
        The measured variation, reduced to *matrix*.

        Blocks with too little to go on take the median of the ones that had
        enough. Saying "typical" there is more honest than either believing a
        spread taken from three voxels or declaring the region silent - noise
        exists outside the mask, it simply was not recorded.
        """
        matrix = tuple(int(n) for n in matrix)
        blocks = self._block_std(self._read(), matrix)

        measured = blocks[np.isfinite(blocks)]
        if measured.size == 0:
            raise ValueError(
                f"No block of {self.path} was {self.min_present:.0%} unmasked. "
                f"Either the volume is empty or the grid asked for is too fine."
            )
        return self._unit_mean(np.where(np.isfinite(blocks), blocks,
                                        np.median(measured)))

    def _read(self) -> np.ndarray:
        """The noise volume, read once."""
        if self._volume is None:
            try:
                import nibabel as nib
            except ImportError as error:
                raise ImportError(
                    "Reading a noise scan needs nibabel."
                ) from error
            self._volume = np.asanyarray(nib.load(self.path).dataobj).astype(np.float64)
        return self._volume

    def _block_std(self, volume: np.ndarray, matrix) -> np.ndarray:
        """
        Spread within each block, ignoring voxels that were masked out.

        Args:
            volume: The noise scan, at its own resolution.
            matrix: Grid to reduce to.

        Returns:
            One standard deviation per block, "(X, Y, Z)", NaN where there was
            not enough left after masking to measure one.
        """
        out = np.full(matrix, np.nan, np.float64)
        edges = [np.linspace(0, volume.shape[axis], n + 1).astype(int)
                 for axis, n in enumerate(matrix[:volume.ndim])]

        for i in range(matrix[0]):
            for j in range(matrix[1]):
                for k in range(matrix[2]):
                    block = volume[edges[0][i]:edges[0][i + 1],
                                   edges[1][j]:edges[1][j + 1],
                                   edges[2][k]:edges[2][k + 1]]
                    seen = block[block > 0]
                    if block.size == 0 or seen.size < self.min_present * block.size:
                        continue
                    spread = seen.std()

                    # The scan is coarsely quantized, so a block can hold enough
                    # voxels and still have them all equal. That is a quirk of
                    # the storage, not a silent region.
                    if spread > 0:
                        out[i, j, k] = spread
        return out


#**************************************************************************************************#
#                                          Class Noise                                             #
#**************************************************************************************************#
#                                                                                                  #
# Adds acquisition noise, working out the right kind from where the data is.                       #
#                                                                                                  #
#**************************************************************************************************#
class Noise(BaseModule):
    """
    Add uncorrelated complex Gaussian noise (AWGN) to MRS data.

    **Backend-agnostic** (zea pattern): "process_tensor" works transparently
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
        Sigma as fraction of max|FID| (alternative to sigma).
        Note that "sigma" and "sigma_frac" are referenced to the data as it
        reaches this module: the spectral transform is non-unitary, so the
        same value means a different absolute noise level in time and
        frequency domain.
    seed : int, optional
        Random seed for reproducibility
    global_scale : bool, default False
        Controls how the reference statistic for "snr" / "snr_db" /
        "sigma_frac" is reduced.  "False" (default) measures it **per FID
        trace**, which is the right thing for SVS.  "True" measures a single
        statistic **per batch element**, reduced over every other axis.

        Use "global_scale=True" for MRSI.  A per-trace statistic on a
        "(B, X, Y, Z, T)" volume gives every voxel its own sigma, so
        background voxels (whose signal power is ~0) receive almost no noise
        while brain voxels receive plenty.  The resulting noise field traces the
        anatomy and hands a network a free brain mask.  "sigma" is an absolute
        scale and is unaffected by this flag.

    Notes
    -----
    Provide EITHER snr_db, sigma, OR sigma_frac (not multiple).
    If snr_db is used, sigma is computed from mean(|fid|^2).

    Examples
    --------
    >>> noise = Noise(snr=20.0)
    >>> noise = Noise(snr_db=10.0)
    >>> noise = Noise(sigma=0.01)
    >>> noise = Noise(sigma_frac=0.02)
    >>> noise = Noise(sigma_frac=0.02, global_scale=True)  # MRSI volumes
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    def __init__(self,
                 covariance: Optional['NoiseCovariance'] = None,
                 profile: Optional['NoiseProfile'] = None,
                 snr: Optional[float] = None,
                 snr_db: Optional[float] = None,
                 sigma: Optional[float] = None,
                 sigma_frac: Optional[float] = None,
                 seed: Optional[int] = None,
                 global_scale: bool = False):
        super().__init__()

        self.covariance = covariance or Independent()
        self.profile = profile or Flat()
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

        Both the scale statistics and the noise itself are computed on the
        tensor's own backend, so the data is never converted and the noise is
        created directly on its device. Randomness comes from
        "nifti_mrs_plus.ops.SeedGenerator": a seeded run is reproducible while
        still drawing fresh noise for every batch.

        Args:
            data_array: Input tensor of shape "(batch, ..., n_points)"
            water_array: Optional water reference tensor (unchanged)
            backend: Backend enum (unused — ops dispatch on the tensor)

        Returns:
            Tuple of (noisy_data, water_array)
        """
        state = kwargs.get('state')
        if (state is not None and state.spatial == 'image'
                and state.sampling == 'undersampled'):
            return self._via_kspace(data_array, water_array, **kwargs)

        original_shape = tuple(data_array.shape)
        ndim = len(original_shape)

        # ── Step 1: noise scale, on the data's own backend ────────────────────
        # `keepdims=True` throughout, so scale always broadcasts against
        # original_shape without any further reshaping.  Reducing to a flat
        # (n_batch, 1) and reshaping only broadcasts correctly when every axis
        # between the batch and the points axis is singleton — true for SVS
        # (B, 1, 1, 1, N), false for MRSI (B, X, Y, Z, T).
        magnitude = ops.abs(data_array)

        if self.sigma is not None:
            # Fixed sigma — no data stats needed, but still built on this
            # backend so the multiply below never crosses frameworks.
            scale = ops.cast_like(magnitude * 0.0 + float(self.sigma), magnitude)
        else:
            # global_scale: one statistic per batch element; otherwise per trace.
            # Per trace means along the spectral axis — axis 4 in the NIfTI
            # layout, the last axis only when no higher dims trail it — never
            # across a coil/average dimension sitting at the end.
            spectral_axis = 4 if ndim > 4 else ndim - 1
            red_axes = tuple(range(1, ndim)) if self.global_scale else (spectral_axis,)

            if self.sigma_frac is not None:
                peak = ops.amax(magnitude, axis=red_axes, keepdims=True)
                peak = ops.where(peak > 0, peak, ops.cast_like(peak * 0.0 + 1.0, peak))
                scale = self.sigma_frac * peak
            else:
                sig_pow = ops.mean(magnitude ** 2, axis=red_axes, keepdims=True) + 1e-16
                if self.snr is not None:
                    noise_pow = sig_pow / float(self.snr)
                else:
                    noise_pow = sig_pow / (10.0 ** (float(self.snr_db) / 10.0))
                scale = ops.sqrt(noise_pow / 2.0)  # per-channel std

        scale = self._shaped(scale, original_shape)

        # ── Step 2: noise, on the data's own backend and device ───────────────
        if not ops.is_complex(data_array):
            # Real data is a magnitude, and the magnitude of a complex signal in
            # complex Gaussian noise is Rice-distributed - non-central chi once
            # coils have been combined. Adding a symmetric perturbation here
            # would let it go negative, which no magnitude ever does.
            real = self.rng.normal(original_shape, like=magnitude)
            imag = self.rng.normal(original_shape, like=magnitude)
            widened = ops.cast_like(scale, real)
            return ops.sqrt((data_array + real * widened) ** 2
                            + (imag * widened) ** 2), water_array

        real = self.rng.normal(original_shape, like=magnitude)
        imag = self.rng.normal(original_shape, like=magnitude)
        noise = ops.complex_from(real * ops.cast_like(scale, real),
                                 imag * ops.cast_like(scale, imag))
        noise = self._correlate(noise, kwargs.get('dim_tags'))

        return data_array + ops.cast_like(noise, data_array), water_array

    #********************#
    #   how loud where   #
    #********************#
    def _shaped(self, scale, shape):
        """
        Modulate the level by where in the volume it is.

        A flat profile is the common case and costs nothing, so it is skipped
        rather than multiplied by ones. Anything else needs the data to have
        real spatial extent - a single-voxel spectrum has nowhere for the level
        to vary - so it is left alone there too.

        Args:
            scale: The level the SNR or sigma asked for.
            shape: The data's shape, NIfTI layout.

        Returns:
            The level, varying across the volume.
        """
        matrix = tuple(int(n) for n in shape[1:4])
        if isinstance(self.profile, Flat) or len(shape) < 5 or max(matrix) == 1:
            return scale

        profile = self.profile.sigma(matrix)

        # open a batch axis in front and one axis per trailing dimension
        view = (1,) + matrix + (1,) * (len(shape) - 4)
        return scale * ops.cast_like(
            ops.match_backend(profile.reshape(view), ops.real(scale)), scale)

    #************************#
    #   where noise enters   #
    #************************#
    def _via_kspace(self, data_array, water_array, **kwargs):
        """
        Add the noise where the scanner would have picked it up.

        Noise enters at the receiver, so it is white in k-space. That is also
        true in the image domain while the data is fully sampled, because the
        transform between them is orthonormal - which is why this module can
        otherwise be dropped anywhere.

        Once k-space has been undersampled the equivalence breaks. A zero-filled
        reconstruction spreads each missing sample over the whole image, so its
        noise is correlated and adding white noise there would not resemble
        anything a scanner produces. The data is therefore taken back to
        k-space, given its noise, and returned.

        Args:
            data_array: Image-domain data that has already been undersampled.
            water_array: Passed through unchanged.

        Returns:
            "(noisy_data, water_unchanged)".
        """
        axes = (1, 2, 3)
        kspace = ops.fftshift(
            ops.fftn(ops.ifftshift(data_array, axis=axes), axes, norm='ortho'), axis=axes)

        # Say so plainly rather than let it look like ordinary image-domain noise
        moved = dict(kwargs)
        moved['state'] = kwargs['state'].having(spatial='kspace')
        noisy, water_array = self.process_tensor(kspace, water_array, **moved)

        return ops.fftshift(
            ops.ifftn(ops.ifftshift(noisy, axis=axes), axes, norm='ortho'),
            axis=axes), water_array

    #*******************#
    #   coil coupling   #
    #*******************#
    def _correlate(self, noise, dim_tags):
        """
        Give the channels the covariance the array actually has.

        Independent draws are mixed by the Cholesky factor of psi, which is the
        standard way to turn white noise into noise with a given covariance.
        Data without a coil axis has nothing to correlate and is left alone.

        Args:
            noise: White complex noise, the shape of the data.
            dim_tags: Higher-dimension tags, to find the coil axis.

        Returns:
            The noise, correlated across channels.
        """
        tags = list(dim_tags or ())
        if isinstance(self.covariance, Independent) or 'DIM_COIL' not in tags:
            return noise

        axis = 5 + tags.index('DIM_COIL')
        shape = ops.shape(noise)
        if axis >= len(shape):
            return noise

        n_coils = int(shape[axis])
        factor = np.linalg.cholesky(
            self.covariance.matrix(n_coils)
            + 1e-8 * np.eye(n_coils, dtype=np.complex64))

        # Mix along the coil axis. The factor is lower triangular, so channel i
        # is a combination of the first i+1 white channels and nothing more.
        columns = []
        for i in range(n_coils):
            mixed = None
            for j in range(i + 1):
                weight = complex(factor[i, j])
                if weight == 0:
                    continue
                term = ops.take(noise, np.array([j]), axis=axis) * ops.cast_like(
                    ops.match_backend(np.array(weight, np.complex64), noise), noise)
                mixed = term if mixed is None else mixed + term
            columns.append(mixed)

        return ops.concatenate(columns, axis=axis)

    #********************************************************************#
    #   pure-numpy path for process_nifti_list (input is always numpy)   #
    #********************************************************************#
    def _add_noise_numpy(self, fid: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to numpy FID data."""
        # Draw from the module's own stream, so this path matches the tensor
        # path: it advances between calls and reproduces from the same seed.
        rng = self.rng.numpy_rng()

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
