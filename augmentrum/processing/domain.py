####################################################################################################
#                                           domain.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Which domain a module works in, and moving data between domains. MRSI transforms along  #
#          two independent axes - spectral and spatial - so a module constrains whichever it       #
#          needs and leaves the other alone.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from dataclasses import dataclass
from typing import Optional

from nifti_mrs_plus import ops

# own
from augmentrum.core import Backend
from augmentrum.core.base_module import BaseModule


__all__ = ['Domain', 'DomainError', 'DomainTransform']


#**************************************************************************************************#
#                                          Class Domain                                            #
#**************************************************************************************************#
#                                                                                                  #
# A domain a module needs, on either axis or both.                                                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Domain:
    """
    A domain a module needs, on either axis or both.

    "None" on an axis means the module does not care, which is the usual case:
    a spectral operation has no opinion about whether the data is in image space
    or k-space, and constraining what it does not use would force transforms
    nobody asked for.

    Attributes:
        spectral: "time", "frequency", or None for no opinion.
        spatial: "image", "kspace", or None for no opinion.
    """

    spectral: Optional[str] = None
    spatial: Optional[str] = None

    def satisfied_by(self, state) -> bool:
        """Whether *state* already sits where this wants it."""
        return ((self.spectral is None or self.spectral == state.spectral)
                and (self.spatial is None or self.spatial == state.spatial))

    def __str__(self) -> str:
        wanted = [f"{axis}={value}" for axis, value
                  in (('spectral', self.spectral), ('spatial', self.spatial)) if value]
        return ', '.join(wanted) or 'any domain'


#**************************************************************************************************#
#                                       Class DomainError                                          #
#**************************************************************************************************#
#                                                                                                  #
# Raised when a module is handed data in a domain it refuses to move itself.                       #
#                                                                                                  #
#**************************************************************************************************#
class DomainError(ValueError):
    """Raised when a module is handed data in a domain it refuses to move itself."""


#**************************************************************************************************#
#                                     Class DomainTransform                                        #
#**************************************************************************************************#
#                                                                                                  #
# Moves data between domains, on either axis and in either direction.                              #
#                                                                                                  #
#**************************************************************************************************#
class DomainTransform(BaseModule):
    """
    Moves data between domains, on either axis and in either direction.

    One module rather than four, because going each way is the same transform
    with a sign. A pipeline inserts these itself where a module needs a domain
    it is not in; a caller adds one deliberately to stay somewhere the default
    would not, such as remaining in k-space so that spatial augmentation acts
    there.

    Each axis keeps the convention already in use elsewhere in the package,
    rather than adopting a single tidy one, because changing either would
    silently rescale every existing amplitude parameter:

    - **spectral** follows the MRS convention, "fftshift(ifft(fid))" forward,
      which is what the spectral modules have always used. It is not unitary -
      it scales by 1/sqrt(N) - so an absolute amplitude means something
      different either side of it. Anything expressed relative to the data,
      such as an SNR, is unaffected because signal and noise scale alike.
    - **spatial** is orthonormal, matching the k-space path, so it preserves
      magnitudes and with them the statistics of noise.

    Args:
        spectral: "time" or "frequency" to move the spectral axis there.
        spatial: "image" or "kspace" to move the spatial axes there.
        seed: Unused; accepted so every module takes one.

    Examples:
        >>> DomainTransform(spatial='kspace')       # image -> k-space
        >>> DomainTransform(spectral='frequency')   # FID -> spectrum
    """

    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    #: Which axis of a batched array each domain lives on.
    SPECTRAL_AXIS = 4
    SPATIAL_AXES = (1, 2, 3)

    def __init__(self, spectral: Optional[str] = None, spatial: Optional[str] = None,
                 seed=None):
        super().__init__()

        for axis, value, allowed in (('spectral', spectral, ('time', 'frequency')),
                                     ('spatial', spatial, ('image', 'kspace'))):
            if value is not None and value not in allowed:
                raise ValueError(f"{axis} must be one of {allowed}, got {value!r}.")
        if spectral is None and spatial is None:
            raise ValueError("DomainTransform needs a spectral or a spatial target.")

        self.target = Domain(spectral=spectral, spatial=spatial)

    def output_state(self, state):
        """The data is now where this transform was asked to put it."""
        state = super().output_state(state)
        changes = {axis: value for axis, value
                   in (('spectral', self.target.spectral), ('spatial', self.target.spatial))
                   if value is not None}
        return state.having(**changes)

    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.NUMPY, **kwargs):
        """
        Move the data to this transform's target domain.

        Args:
            data_array: "(batch, X, Y, Z, T, ...)" complex, on any backend.
            water_array: Passed through unchanged.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Absorbs what BaseModule injects, and reads "state" from it
                to know which way to go.

        Returns:
            "(transformed_data, water_unchanged)".
        """
        state = kwargs.get('state')
        if state is None:
            raise DomainError(
                "DomainTransform needs to know where the data is. It is given the "
                "state by BaseModule, so call it on a NIfTI_MRS_Plus rather than "
                "on a bare tensor."
            )

        if self.target.spectral is not None and self.target.spectral != state.spectral:
            data_array = self._spectral(data_array, to=self.target.spectral)
        if self.target.spatial is not None and self.target.spatial != state.spatial:
            data_array = self._spatial(data_array, to=self.target.spatial)

        return data_array, water_array

    #****************#
    #   transforms   #
    #****************#
    def _spectral(self, data_array, to: str):
        """
        FID to spectrum and back, in the MRS convention.

        The backends only transform their last axis, so the spectral one is
        brought there and put back. On the common "(batch, X, Y, Z, T)" that
        costs nothing, since it is already last; it matters once a coil axis
        sits behind it.
        """
        rank = len(ops.shape(data_array))
        axis = self.SPECTRAL_AXIS
        moved = axis != rank - 1

        if moved:
            forward = [d for d in range(rank) if d != axis] + [axis]
            data_array = ops.transpose(data_array, forward)

        if to == 'frequency':
            data_array = ops.fftshift(ops.ifft(data_array), axis=-1)
        else:
            data_array = ops.fft(ops.ifftshift(data_array, axis=-1))

        if moved:
            back = list(range(axis)) + [rank - 1] + list(range(axis, rank - 1))
            data_array = ops.transpose(data_array, back)
        return data_array

    def _spatial(self, data_array, to: str):
        """Image to k-space and back, orthonormal so magnitudes are preserved."""
        axes = self.SPATIAL_AXES
        if to == 'kspace':
            return ops.fftshift(ops.fftn(ops.ifftshift(data_array, axis=axes), axes,
                                         norm='ortho'), axis=axes)
        return ops.fftshift(ops.ifftn(ops.ifftshift(data_array, axis=axes), axes,
                                      norm='ortho'), axis=axes)
