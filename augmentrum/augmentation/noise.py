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
import warnings
import numpy as np
from typing import Optional, List

from abc import ABC, abstractmethod

from augmentrum.core.base_module import BaseModule
from augmentrum.core import precision as prec
from augmentrum.processing.utils import to_backend
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
        return (psi / scale[:, None] / scale[None, :]).astype(np.complex128)


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
        return np.eye(n_coils, dtype=np.complex128)


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
        return (profile / mean if mean > 0 else np.ones_like(profile)).astype(np.float64)


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
        return np.ones(tuple(int(n) for n in matrix), np.float64)


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
        self.profile = np.asarray(profile, dtype=np.float64)

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
    Add complex Gaussian acquisition noise (AWGN) to MRS data.

    Thermal noise enters at the receiver: white, complex Gaussian, in the time
    domain. This module can be dropped anywhere in a pipeline - time or
    frequency domain, image space or k-space, on raw coils or on combined data,
    SVS or MRSI - and adds that same noise wherever it finds the data, so a
    level means one thing regardless of placement.

    **The level is defined in MRS terms and is domain-invariant.** "The
    spectrum" is the unitary DFT of the FID, "fftshift(fft(fid, norm='ortho'))"
    - the normalisation FSL-MRS reports SNR in and
    :func:"augmentrum.processing.utils.fid_to_spec" uses, less that function's
    half-first-point baseline correction, which shifts the peak by well under a
    percent. Under it, white time-domain noise of per-channel SD sigma has
    per-channel SD sigma in the spectrum as well. So:

    - "sigma" is the SD of the added noise per point and per channel (real and
      imaginary each) in the time-domain FID, which equals its SD in the
      unitary spectrum;
    - "snr" is the peak SNR the added noise alone gives the trace,
      "max|spectrum| / sigma" - peak height over the SD of the real-part
      noise, the way MRS SNR is reported;
    - "snr_db" is the same in decibels, "20 log10(snr)": the amplitude
      convention, because a peak height over a noise SD is a ratio of
      amplitudes rather than of powers;
    - "sigma_frac" is the noise SD as a fraction of the spectrum peak,
      "sigma / max|spectrum|", which is "1 / snr".

    The reference peak is measured on the data as it reaches the module, and
    the noise already in the data is not subtracted: the parameter describes
    the noise *added*. A spectrum with peak SNR 120 that receives "snr=20"
    ends up near "1 / sqrt(1/120**2 + 1/20**2) = 19.7".

    Placement decides what a level means physically, and that is intended.
    Noise added per coil and per transient, before combination and averaging,
    gives each raw trace the requested SNR; the combined spectrum then gains
    roughly "sqrt(N_averages)" from averaging and the array's combination gain
    on top, exactly as it does at the scanner. The module does not compensate
    for that.

    **Domain awareness.** In the time domain the reference peak comes from one
    batched FFT of the data. In the frequency domain, which a pipeline reaches
    through :class:"DomainTransform"'s "fftshift(ifft(fid))" (a "1/N"
    normalisation), the peak is read off the data and "sigma" is scaled by
    "1/sqrt(N)", so the same noise sits on the FID after the transform back.
    In k-space the reference comes from the image the data transforms to: the
    spatial transform is orthonormal and preserves the noise level but not the
    peak. Undersampled data is taken to k-space, where receiver noise is white,
    noised there and returned. Real (magnitude) data receives Rician noise.

    **Backend-agnostic**: "process_tensor" works on NumPy, PyTorch, JAX and
    TensorFlow tensors, with the noise drawn on the data's own device.

    Args:
        covariance: How the channels of a receive array share their noise;
            :class:"Independent" by default.
        profile: How loud the noise is from place to place across a volume;
            :class:"Flat" by default. An image-domain description, so it is
            applied in image space only.
        snr: Peak SNR of the added noise, "max|spectrum| / sigma".
        snr_db: The same in dB, "20 log10(snr)".
        sigma: SD of the added noise per point and per channel, time domain.
        sigma_frac: Noise SD as a fraction of the spectrum peak, "1 / snr".
        seed: Random seed for reproducibility.
        global_scale: How the reference peak for "snr" / "snr_db" /
            "sigma_frac" is reduced. "False" measures it per trace; "True"
            once per batch element, over every voxel, coil and transient it
            holds. "None" (default) picks "True" whenever a batch element holds
            more than one trace, "False" otherwise.

            One reference per batch element is the physical choice for coil
            arrays and MRSI. The noise level is a property of the receiver,
            not of what each channel or voxel happens to see, so a per-trace
            reference would give a far coil - or a background voxel - almost
            no noise, and hand a network a free sensitivity map or brain mask.
            "sigma" is absolute and unaffected by this flag.

    Notes:
        Provide exactly one of "snr", "snr_db", "sigma" or "sigma_frac".

    Examples:
        >>> noise = Noise(snr=20.0)                     # added noise alone: peak SNR 20
        >>> noise = Noise(snr_db=26.0)                  # the same, in dB
        >>> noise = Noise(sigma=0.01)                   # absolute time-domain SD
        >>> noise = Noise(sigma_frac=0.05)              # noise SD 5 % of the peak
        >>> noise = Noise(snr=5.0, global_scale=False)  # per trace, regardless
    """

    SUPPORTED_BACKENDS = tuple(Backend)

    # The level broadcasts, so a batch can carry one SNR / sigma per sample.
    PER_SAMPLE_PARAMS = ('snr', 'snr_db', 'sigma', 'sigma_frac')

    #: The spatial axes of a batched array, for the k-space paths.
    SPATIAL_AXES = (1, 2, 3)

    def __init__(self,
                 covariance: Optional['NoiseCovariance'] = None,
                 profile: Optional['NoiseProfile'] = None,
                 snr: Optional[float] = None,
                 snr_db: Optional[float] = None,
                 sigma: Optional[float] = None,
                 sigma_frac: Optional[float] = None,
                 seed: Optional[int] = None,
                 global_scale: Optional[bool] = None):
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
        """
        Add noise to a list of NIFTI_MRS objects.

        One object is one batch element, so it goes through the same steps as
        the tensor path with a batch axis in front - the level means the same
        on either engine. A per-sample vector is read at this subject's index.
        """
        state = kwargs.get('state')
        processed_data = []
        for i, nifti in enumerate(data_list):
            batch = np.asarray(nifti[:])[None]
            sigma, snr = self._requested(index=i)
            scale = self._shaped(self._scale(batch, sigma, snr, state), batch.shape, state)
            nifti[:] = self._add(batch, scale, nifti.dim_tags)[0]
            processed_data.append(nifti)
        return processed_data, water_list

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add noise to tensor/array data (**any backend, natively**).

        Both the reference statistic and the noise itself are computed on the
        tensor's own backend, so the data is never converted and the noise is
        created directly on its device. Randomness comes from
        "nifti_mrs_plus.random.SeedGenerator": a seeded run is reproducible on
        a given backend while still drawing fresh noise for every batch.

        Args:
            data_array: Input tensor in the NIfTI layout "(batch, X, Y, Z, T, ...)".
            water_array: Optional water reference tensor (unchanged).
            backend: Backend enum (unused - ops dispatch on the tensor).
            **kwargs: What BaseModule injects; "state" says which domain the
                data is in and "dim_tags" where its coil axis sits.

        Returns:
            Tuple of (noisy_data, water_array).
        """
        state = kwargs.get('state')
        dim_tags = kwargs.get('dim_tags')
        sigma, snr = self._requested()

        if (state is not None and state.spatial == 'image'
                and state.sampling == 'undersampled'):
            return self._via_kspace(data_array, sigma, snr, state, dim_tags), water_array

        scale = self._shaped(self._scale(data_array, sigma, snr, state),
                             ops.shape(data_array), state)
        return self._add(data_array, scale, dim_tags), water_array

    #**************#
    #   how loud   #
    #**************#
    def _requested(self, index=None):
        """
        The level as "(sigma, snr)", exactly one of them set.

        Four ways of stating a level reduce to two: an absolute SD, or a peak
        SNR that needs a reference. "snr_db" and "sigma_frac" are conversions
        of "snr". A per-sample vector is kept whole for the tensor path and
        read at one subject's *index* for the list path.

        Args:
            index: Which subject's value to pick out of a per-sample vector,
                or None to keep the vector.

        Returns:
            "(sigma, None)" or "(None, snr)", as float64 arrays.
        """
        pick = (lambda v: v) if index is None else (lambda v: self.sample_of(v, index))
        if self.sigma is not None:
            return np.asarray(pick(self.sigma), dtype=np.float64), None
        if self.snr is not None:
            return None, np.asarray(pick(self.snr), dtype=np.float64)
        if self.snr_db is not None:
            return None, 10.0 ** (np.asarray(pick(self.snr_db), dtype=np.float64) / 20.0)
        return None, 1.0 / np.asarray(pick(self.sigma_frac), dtype=np.float64)

    def _scale(self, data_array, sigma, snr, state, force_global: bool = False):
        """
        The per-channel SD of the noise to add, in the data's own domain.

        Every domain the data can be in is handled here, so that the same
        request adds the same noise wherever the module sits. A "sigma" is a
        time-domain quantity: in the frequency domain it is divided by
        "sqrt(N)", because the "1/N" transform the pipeline uses shrinks white
        noise by exactly that. An SNR needs no such correction, since the peak
        it is relative to is measured in the same domain as the noise is added.

        Args:
            data_array: The data, NIfTI layout.
            sigma: Absolute level, or None.
            snr: Peak SNR, or None.
            state: Where the data is; None means the canonical time domain.
            force_global: Reduce the reference per batch element regardless of
                "global_scale" - for noise added in k-space, which is one level
                per batch element by definition.

        Returns:
            The SD, shaped to broadcast against the data.
        """
        shape = ops.shape(data_array)
        ndim = len(shape)
        axis = self.SPECTRAL_AXIS if ndim > self.SPECTRAL_AXIS + 1 else ndim - 1
        in_frequency = state is not None and state.spectral == 'frequency'

        if sigma is not None:
            level = self._level(sigma, data_array, ndim)
            return level / float(np.sqrt(shape[axis])) if in_frequency else level

        # One reference per batch element whenever there is more than one
        # trace to share it, unless the caller decided otherwise.
        global_scale = self.global_scale
        if global_scale is None:
            global_scale = int(np.prod(shape[1:axis] + shape[axis + 1:])) > 1

        peak = self._peak(data_array, state, global_scale or force_global)
        return peak / ops.cast_like(self._level(snr, peak, ndim), peak)

    def _peak(self, data_array, state, global_scale: bool):
        """
        The spectrum peak an SNR is relative to, in the data's own domain.

        In the time domain that takes one batched FFT; the unnormalised
        transform is divided by "sqrt(N)" to give the unitary spectrum the
        level is defined in. In the frequency domain the data *is* the
        spectrum, in whatever normalisation the pipeline used - the ratio to
        the noise added in the same domain does not depend on it. In k-space
        the peak is read from the image the data transforms to, because the
        orthonormal spatial transform preserves the noise level but not the
        peak, which is an image-domain quantity.

        Args:
            data_array: The data, NIfTI layout.
            state: Where the data is; None means time domain, image space.
            global_scale: One peak per batch element rather than per trace.

        Returns:
            "max|spectrum|", with kept dimensions so it broadcasts.
        """
        shape = ops.shape(data_array)
        ndim = len(shape)
        axis = self.SPECTRAL_AXIS if ndim > self.SPECTRAL_AXIS + 1 else ndim - 1
        axes = self.SPATIAL_AXES

        x = data_array
        if (state is not None and state.spatial == 'kspace'
                and ndim > axis and max(shape[1:4]) > 1):
            x = ops.fftshift(ops.ifftn(ops.ifftshift(x, axis=axes), axes, norm='ortho'),
                             axis=axes)

        if state is not None and state.spectral == 'frequency':
            peak = ops.amax(ops.abs(x), axis=axis, keepdims=True)
        else:
            # The backends transform their last axis only, so the spectral one
            # is brought there when a coil or average axis sits behind it.
            moved = axis != ndim - 1
            if moved:
                x = ops.transpose(x, [d for d in range(ndim) if d != axis] + [axis])
            if not ops.is_complex(x):
                x = ops.complex_from(x, x * 0.0)     # tf.signal.fft takes complex only
            peak = ops.amax(ops.abs(ops.fft(x)), axis=-1, keepdims=True)
            if moved:
                peak = ops.transpose(peak, list(range(axis)) + [ndim - 1]
                                     + list(range(axis, ndim - 1)))
            peak = peak / float(np.sqrt(shape[axis]))

        if global_scale and ndim > 1:
            peak = ops.amax(peak, axis=tuple(range(1, ndim)), keepdims=True)

        # A silent trace has nothing to be relative to; a unit peak keeps the
        # division finite rather than producing NaN.
        return ops.where(peak > 0, peak, ops.cast_like(peak * 0.0 + 1.0, peak))

    @staticmethod
    def _level(value, like, ndim):
        """
        A level parameter as a tensor that broadcasts against the data.

        A scalar becomes a single element, a "(batch,)" per-sample vector a
        column, both on *like*'s backend and device, so nothing downstream
        crosses frameworks.

        Args:
            value: A scalar, or a "(batch,)" vector of per-sample values.
            like: Any tensor on the target backend.
            ndim: Rank of the data the level will multiply.

        Returns:
            A tensor of shape "(1 or batch, 1, ..., 1)", real, in *like*'s precision.
        """
        arr = np.asarray(value, dtype=np.float64).reshape((-1,) + (1,) * (ndim - 1))
        return to_backend(arr, like, dtype=prec.real_name(like))

    #***************#
    #   how where   #
    #***************#
    def _shaped(self, scale, shape, state):
        """
        Modulate the level by where in the volume it is.

        A flat profile is the common case and costs nothing, so it is skipped
        rather than multiplied by ones. Anything else needs the data to have
        real spatial extent - a single-voxel spectrum has nowhere for the level
        to vary - and to be in image space, where position means something; in
        k-space the noise is white receiver noise and a profile cannot apply.

        Args:
            scale: The level the SNR or sigma asked for.
            shape: The data's shape, NIfTI layout.
            state: Where the data is; None means image space.

        Returns:
            The level, varying across the volume.
        """
        matrix = tuple(int(n) for n in shape[1:4])
        if isinstance(self.profile, Flat) or len(shape) < 5 or max(matrix) == 1:
            return scale
        if state is not None and state.spatial == 'kspace':
            warnings.warn(
                f"{self.profile.__class__.__name__} describes the noise level across the "
                f"image, but the data is in k-space, where the noise is white by nature. "
                f"It is added flat here; place Noise in image space for the profile to apply.",
                RuntimeWarning)
            return scale

        profile = self.profile.sigma(matrix)

        # open a batch axis in front and one axis per trailing dimension
        view = (1,) + matrix + (1,) * (len(shape) - 4)
        return scale * ops.match_backend(profile.reshape(view), scale)

    #************************#
    #   where noise enters   #
    #************************#
    def _add(self, data_array, scale, dim_tags):
        """
        Draw the noise at the given level and add it, on the data's own device.

        Args:
            data_array: The data, any backend.
            scale: Per-channel SD, broadcastable against the data.
            dim_tags: Higher-dimension tags, to find the coil axis to correlate.

        Returns:
            The data with its noise.
        """
        shape = ops.shape(data_array)
        real = self.rng.normal(shape, like=data_array, dtype=prec.real_name(data_array))
        imag = self.rng.normal(shape, like=data_array, dtype=prec.real_name(data_array))
        widened = ops.cast_like(scale, real)

        if not ops.is_complex(data_array):
            # Real data is a magnitude, and the magnitude of a complex signal in
            # complex Gaussian noise is Rice-distributed - non-central chi once
            # coils have been combined. Adding a symmetric perturbation here
            # would let it go negative, which no magnitude ever does.
            return ops.sqrt((data_array + ops.cast_like(real * widened, data_array)) ** 2
                            + ops.cast_like(imag * widened, data_array) ** 2)

        noise = ops.complex_from(real * widened, imag * widened)
        noise = self._correlate(noise, dim_tags)
        return data_array + ops.cast_like(noise, data_array)

    def _via_kspace(self, data_array, sigma, snr, state, dim_tags):
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
        k-space, given its noise, and returned. The level is still referenced
        to the image, where a spectrum peak means something, and is one value
        per batch element because white noise has no position.

        Args:
            data_array: Image-domain data that has already been undersampled.
            sigma: Absolute level, or None.
            snr: Peak SNR, or None.
            state: The (image-domain) state the data arrived in.
            dim_tags: Higher-dimension tags, to find the coil axis.

        Returns:
            The noisy data, back in the image domain.
        """
        scale = self._scale(data_array, sigma, snr, state, force_global=True)

        axes = self.SPATIAL_AXES
        kspace = ops.fftshift(
            ops.fftn(ops.ifftshift(data_array, axis=axes), axes, norm='ortho'), axis=axes)
        noisy = self._add(kspace, scale, dim_tags)

        return ops.fftshift(
            ops.ifftn(ops.ifftshift(noisy, axis=axes), axes, norm='ortho'), axis=axes)

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
            + 1e-8 * np.eye(n_coils, dtype=np.complex128))

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
                    ops.match_backend(np.array(weight, np.complex128), noise), noise)
                mixed = term if mixed is None else mixed + term
            columns.append(mixed)

        return ops.concatenate(columns, axis=axis)
