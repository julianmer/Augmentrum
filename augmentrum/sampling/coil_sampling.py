####################################################################################################
#                                        coil_sampling.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-07                                                                              #
#                                                                                                  #
# Purpose: Everything that shapes a receive array: synthesizing one from sensitivity maps, and     #
#          drawing from the coils that are there. Sensitivities encode position, so giving         #
#          coil-combined data an array is what turns undersampling into parallel imaging rather    #
#          than plain decimation.                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from nifti_mrs_plus import ops

# own
from abc import ABC, abstractmethod

from augmentrum.core import Backend
from augmentrum.core.base_module import BaseModule
from augmentrum.utils import download
from augmentrum.sampling.dimension_sampling import DimensionSampler


__all__ = ['MapSource', 'Birdcage', 'Supplied', 'T2starMove', 'CoilSampler']


#**************************************************************************************************#
#                                         Class MapSource                                          #
#**************************************************************************************************#
#                                                                                                  #
# Where sensitivity maps come from.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class MapSource(ABC):
    """
    Where sensitivity maps come from.

    Synthesizing an array, using one someone measured, and downloading one from
    a public dataset are the same job answered three ways, and a caller usually
    wants to swap between them without anything else changing. Stating it as a
    contract keeps each source's own settings - a ring radius, a subject
    identifier - on the source rather than piling them onto whatever uses it.

    A source produces its natural number of elements; resizing to a different
    count is done here, once, so no implementation has to think about it.
    """

    def maps(self, matrix, n_coils=None, rng=None) -> np.ndarray:
        """
        Sensitivity maps over a spatial grid.

        Args:
            matrix: Grid to cover, "(X, Y, Z)".
            n_coils: Elements wanted. "None" takes whatever the source has.
                Fewer are compressed onto the leading singular vectors; more
                are made by displacing the ones there, which adds elements but
                not information - see :func:"extend_array".
            rng: NumPy generator, for sources that place elements at random and
                for extending.

        Returns:
            Maps "(X, Y, Z, C)", root-sum-of-squares 1 wherever the array sees
            anything.
        """
        matrix = tuple(int(n) for n in matrix)
        maps = self.build(matrix, n_coils, rng)

        have = maps.shape[-1]
        if n_coils is None or n_coils == have:
            return maps
        if n_coils < have:
            return self.compress(maps, n_coils)
        return self.extend(maps, n_coils, rng or np.random.default_rng())

    @abstractmethod
    def build(self, matrix, n_coils, rng) -> np.ndarray:
        """
        This source's own maps over *matrix*, before any resizing.

        A source that can honour *n_coils* directly should; one that cannot
        ignores it and lets :meth:"maps" resize what comes back.
        """

    #**************#
    #   resizing   #
    #**************#
    @classmethod
    def compress(cls, maps: np.ndarray, n_coils: int) -> np.ndarray:
        """
        Fewest virtual coils that still see what the array saw.

        Channels of a real array overlap heavily, so most of what it measures
        lives in far fewer directions than it has elements. The leading
        singular vectors across the coil axis are those directions, and
        projecting onto them is what a scanner does when it compresses a
        32-channel acquisition down to eight.

        Args:
            maps: Maps "(X, Y, Z, C)".
            n_coils: Virtual coils to keep. Above the rank of *maps* there is
                nothing left to keep, so the extra ones would be empty.

        Returns:
            Maps "(X, Y, Z, n_coils)", renormalized.
        """
        flat = maps.reshape(-1, maps.shape[-1])
        _, _, basis = np.linalg.svd(flat, full_matrices=False)

        return cls.normalize(np.tensordot(maps, basis[:n_coils].conj().T,
                                          axes=([-1], [0])))

    @classmethod
    def extend(cls, maps: np.ndarray, n_coils: int, rng) -> np.ndarray:
        """
        More elements than were measured, by moving the ones that were.

        Shifting a measured map describes an element sitting somewhere else,
        which is a function the original array does not contain - so the result
        really does have more independent channels, unlike any recombination of
        what is already there.

        What it cannot do is recover information the acquisition never had.
        These are plausible extra elements for simulating a larger array, not
        extra knowledge about the subject that was scanned.

        Args:
            maps: Maps "(X, Y, Z, C)".
            n_coils: Elements wanted, more than *maps* has.
            rng: NumPy generator placing the extra elements.

        Returns:
            Maps "(X, Y, Z, n_coils)", renormalized.
        """
        grown = [maps]
        have = maps.shape[-1]

        while have < n_coils:
            moved = maps
            for axis, extent in enumerate(maps.shape[:3]):
                moved = np.roll(moved, int(rng.integers(1, max(2, extent))), axis=axis)
            grown.append(moved)
            have += maps.shape[-1]

        return cls.normalize(np.concatenate(grown, axis=-1)[..., :n_coils])

    @staticmethod
    def normalize(maps: np.ndarray) -> np.ndarray:
        """
        Unit total sensitivity, leaving voxels the array never saw at zero.

        Every source ends here, so maps from any of them combine the same way
        and can stand in for one another.
        """
        maps = maps.astype(np.complex64)
        rss = np.sqrt((np.abs(maps) ** 2).sum(axis=-1, keepdims=True))
        return np.divide(maps, rss, out=np.zeros_like(maps), where=rss > 0)


#**************************************************************************************************#
#                                         Class Birdcage                                           #
#**************************************************************************************************#
#                                                                                                  #
# A synthetic birdcage array, built from grid geometry alone.                                      #
#                                                                                                  #
#**************************************************************************************************#
class Birdcage(MapSource):
    """
    A synthetic birdcage array, built from grid geometry alone.

    Needs no data and no download, and works at any matrix size, which is what
    makes it the default: every dataset can have a receive array this way.

    Elements sit on one or more rings around the volume.

    Args:

        n_coils: Elements to build when the caller does not say.
        radius: Ring radius in half-FOV units. Above 1 the elements sit outside
            the volume, as a real array does.
        n_rings: Rings stacked along z, so the array encodes z as well as the
            plane. Ignored for a 2-D matrix, where there is no z to encode.

    Examples:
        >>> maps = Birdcage(n_coils=8).maps((32, 32, 8))
        >>> maps.shape
        (32, 32, 8, 8)
        >>> bool(np.allclose(np.sqrt((np.abs(maps) ** 2).sum(-1)), 1.0))
        True
    """

    def __init__(self, n_coils: int = 8, radius: float = 1.5, n_rings: int = 1):
        self.n_coils = int(n_coils)
        self.radius = float(radius)
        self.n_rings = int(n_rings)

    def build(self, matrix, n_coils, rng):
        """
        Build directly at the count asked for, and turn the array a little.

        Each element sees a magnitude falling as 1/r from its own position and
        a phase that winds with the angle to it, which is what lets the coils
        distinguish position: two voxels the same distance from one element
        differ in how the others see them.

        This depends only on grid geometry, never on data, so it is built in
        NumPy and promoted to the caller's backend where it is applied.
        """
        n_coils = int(n_coils or self.n_coils)
        if n_coils < 1:
            raise ValueError(f"n_coils must be at least 1, got {n_coils}.")
        if len(matrix) not in (2, 3):
            raise ValueError(f"Birdcage needs a 2-D or 3-D matrix, got {len(matrix)}-D.")

        angle, positions = self._elements(n_coils, len(matrix))
        offsets = [g[..., None] - p
                   for g, p in zip(self._voxels(matrix), positions)]

        distance = np.sqrt(sum(offset ** 2 for offset in offsets))
        phase = np.arctan2(offsets[0], -offsets[1]) - angle

        # A rotation per call, so a batch does not meet an identically placed
        # array every time while a fixed seed still replays exactly.
        if rng is not None:
            phase = phase + rng.uniform(0.0, 2.0 * np.pi)

        return self.normalize(np.exp(1j * phase) / distance)

    def _elements(self, n_coils, ndim):
        """Where the elements sit: angle around the ring, offset along z."""
        n_rings = 1 if ndim == 2 else max(1, min(self.n_rings, n_coils))
        per_ring = int(np.ceil(n_coils / n_rings))

        coil = np.arange(n_coils)
        angle = 2.0 * np.pi * (coil % per_ring) / per_ring

        # Rings are centered on the volume, so a single one sits in its mid-plane.
        offset = coil // per_ring - 0.5 * (n_rings - 1.0)
        return angle, [self.radius * np.cos(angle), self.radius * np.sin(angle), offset]

    @staticmethod
    def _voxels(matrix):
        """Voxel positions, normalized so the volume spans [-1, 1] on each axis."""
        axes = [(np.arange(n) - n / 2.0) / (n / 2.0) if n > 1 else np.zeros(1)
                for n in matrix]
        grids = list(np.meshgrid(*axes, indexing='ij'))

        # A 2-D grid has no z to place elements against, so it collapses onto zero.
        return grids if len(matrix) == 3 else grids + [np.zeros_like(grids[0])]


#**************************************************************************************************#
#                                         Class Supplied                                           #
#**************************************************************************************************#
#                                                                                                  #
# Maps the caller already has.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class Supplied(MapSource):
    """
    Maps the caller already has.

    Args:
        maps: Sensitivity maps "(X, Y, Z, C)", covering the grid they will be
            applied to.
    """

    def __init__(self, maps):
        self.array = np.asarray(maps)

    def build(self, matrix, n_coils, rng):
        """Hand them over, once they are known to fit."""
        if self.array.shape[:-1] != matrix:
            raise ValueError(
                f"Supplied maps cover {self.array.shape[:-1]} but the data is "
                f"{matrix}. Sensitivity maps must match the spatial matrix."
            )
        return self.array


#**************************************************************************************************#
#                                        Class T2starMove                                          #
#**************************************************************************************************#
#                                                                                                  #
# Maps measured on a real 32-channel head array, from the T2*-MOVE dataset.                        #
#                                                                                                  #
#**************************************************************************************************#
class T2starMove(MapSource):
    """
    Maps measured on a real 32-channel head array, from the T2*-MOVE dataset.

    A real array is not smooth the way a synthetic one is: elements differ in
    loading, coupling and placement, and the maps stop at the edge of the head
    rather than filling the grid. This is the one open dataset that ships
    estimated maps rather than raw data needing calibration.

    The archive is 5.4 GB and the host serves no byte ranges, so there is no way
    to take less. It is read once; the cache it leaves behind is a hundredth of
    that, and every later call reads that instead.

    The data is other people\'s work, released under CC-BY-4.0. It is fetched
    from the publisher rather than redistributed here, and using it means citing
    them - see :attr:"CITATION".

    Args:
        subject: Subject to lift maps from, e.g. "sub-04". Defaults to the
            first in the archive.
        keep_archive: Keep the 5.4 GB download after extracting. Off by
            default, since only the cached maps are needed again.
        progress: Report download progress to stdout.
    """

    URL = ('https://nefeli.helmholtz-munich.de/api/records/9hqjb-tfw77'
           '/files/mr_data-val_recon.zip/content')
    ARCHIVE = 'mr_data-val_recon.zip'
    BYTES = 5362734881
    MD5 = 'c34528703059ce474ea27c5fce646eb8'
    MEMBER = 't2s_gre_fr.hf'                # the raw acquisition, which carries the maps
    DOI = '10.15134/2kek-3553'
    LICENSE = 'CC-BY-4.0'
    CITATION = ('Eichhorn H, Saks E, Spieker VJ, Preibisch C, Schnabel JA. '
                'T2*-MOVE: An Open k-Space Dataset to Address Motion in T2* '
                'Quantification from Gradient Echo MRI (2025). '
                'doi:10.15134/2kek-3553')

    def __init__(self, subject=None, keep_archive: bool = False, progress: bool = True):
        self.subject = subject
        self.keep_archive = keep_archive
        self.progress = progress

    def build(self, matrix, n_coils, rng):
        """The measured maps, downloading the archive the first time."""
        maps = self.fetch(self.subject, self.keep_archive, self.progress)

        if maps.shape[:-1] != matrix:
            raise ValueError(
                f"T2*-MOVE maps cover {maps.shape[:-1]} but the data is {matrix}. "
                f"Resample one to the other before applying them."
            )
        return maps

    @classmethod
    def fetch(cls, subject=None, keep_archive=False, progress=True) -> np.ndarray:
        """
        Maps for one subject, downloading the archive if need be.

        Returns:
            Maps "(X, Y, Z, C)" complex64, root-sum-of-squares 1 wherever the
            array saw signal and 0 outside the head, where a measured map has
            nothing to normalize.

        Raises:
            ImportError: If h5py is not installed.
            RuntimeError: If the download does not match its published checksum.
        """
        import zipfile

        cache = download.cache_root() / 'csm'
        cache.mkdir(parents=True, exist_ok=True)

        cached = cache / f"t2star_move_{subject or 'first'}.npz"
        if cached.exists():
            return np.load(cached)['maps']

        archive = cache / cls.ARCHIVE
        if not archive.exists():
            if progress:
                print(f"Fetching T2*-MOVE sensitivity maps "
                      f"({cls.BYTES / 1e9:.1f} GB, {cls.LICENSE}) from {cls.DOI}.\n"
                      f"  This runs once; the cached maps are a few MB.\n"
                      f"  Please cite: {cls.CITATION}")
            download.fetch(cls.URL, archive, md5=cls.MD5, size=cls.BYTES,
                           progress=progress)

        with zipfile.ZipFile(archive) as bundle:
            member = cls._member(bundle, subject)
            if progress:
                print(f"  extracting {member}")
            raw = cache / 'raw.hf'
            with bundle.open(member) as source, open(raw, 'wb') as target:
                target.write(source.read())

        maps = cls._read(raw)
        raw.unlink()
        np.savez_compressed(cached, maps=maps)

        if not keep_archive:
            archive.unlink()

        return maps

    @classmethod
    def _member(cls, bundle, subject):
        """Path inside *bundle* holding the raw acquisition to read maps from."""
        members = sorted(n for n in bundle.namelist() if n.endswith('/' + cls.MEMBER))
        if not members:
            raise RuntimeError(f"No {cls.MEMBER} inside {cls.ARCHIVE}.")

        if subject is None:
            return members[0]

        for name in members:
            if f'/{subject}/' in name:
                return name
        raise ValueError(
            f"{subject} is not in this archive. It holds: "
            f"{sorted({n.split('/')[-2] for n in members})}"
        )

    @staticmethod
    def _read(path) -> np.ndarray:
        """
        Read "sens_maps" out of a raw file, in this package\'s layout.

        The dataset stores maps as "(slices, echoes, coils, PE, readout)" -
        coils third, as its own loader assumes when it combines over that axis.
        Here they are wanted as "(X, Y, Z, C)".
        """
        try:
            import h5py
        except ImportError as error:
            raise ImportError(
                "Reading measured maps needs h5py. Install it with "
                "\"pip install h5py\", or use Birdcage for synthetic ones."
            ) from error

        with h5py.File(path, 'r') as source:
            maps = np.asarray(source['sens_maps'])

        if maps.ndim == 5:                  # drop the echo axis; maps are shared
            maps = maps[:, 0]
        if maps.ndim != 4:
            raise RuntimeError(f"Unexpected sens_maps rank {maps.ndim}, expected 4 or 5.")

        # (slices, coils, PE, readout) -> (readout, PE, slices, coils)
        return T2starMove.normalize(np.transpose(maps, (3, 2, 0, 1)))


#**************************************************************************************************#
#                                        Class CoilSampler                                         #
#**************************************************************************************************#
#                                                                                                  #
# Shapes a receive array: synthesize one, or draw from the coils already there.                    #
#                                                                                                  #
#**************************************************************************************************#
class CoilSampler(DimensionSampler):
    """
    Shapes a receive array: synthesize one, or draw from the coils already there.

    Most MRSI datasets arrive coil-combined, so undersampling them is plain
    decimation: nothing encodes the discarded positions and nothing can recover
    them. Synthesizing an array puts that encoding back, because each element
    weights the volume differently and so measures a different projection of
    it. Data acquired on a real array already has one, and there the useful
    operation is the opposite - keep fewer elements, and simulate a smaller
    receive array.

    Modes:

    "synthesize"
        Grow a coil axis, "(batch, X, Y, Z, T)" to "(batch, X, Y, Z, T, C)",
        following the NIfTI-MRS convention of coils after the spectral axis.
        Applying maps is a multiply, so gradients reach the data.
    "random"
        Draw a random subset of the coils that are there.
    "deterministic"
        Keep every coil, or exactly the indices passed to the call.

    Averages are a separate acquisition axis and have their own sampler; see
    :class:"augmentrum.sampling.AverageSampler".

    Args:
        mode: "random" or "deterministic" to draw, "synthesize" to build an
            array. Drawing is the default, because data that has coils is the
            common case and synthesis is a deliberate act.
        n_coils: Elements to synthesize, or "(min, max)" coils to draw (default
            all). A bare number means exactly that many. When synthesizing,
            "None" takes however many the source has.
        source: Where maps come from - :class:"Birdcage" by default,
            :class:"Supplied" for maps already in hand, :class:"T2starMove" for
            measured ones. Unused when drawing.
        seed: Fixes the sequence. Draws still vary from call to call.

    Examples:
        >>> import numpy as np
        >>> volume = np.ones((2, 8, 8, 4, 16), np.complex64)
        >>> array = CoilSampler(mode='synthesize', source=Birdcage(n_coils=8))
        >>> with_array, _ = array.process_tensor(volume)
        >>> with_array.shape
        (2, 8, 8, 4, 16, 8)
        >>> drawn, _ = CoilSampler(mode='random', n_coils=3, seed=0).process_tensor(
        ...     with_array, dim_tags=['DIM_COIL', None, None])
        >>> drawn.shape
        (2, 8, 8, 4, 16, 3)
    """

    DIM_TAG = 'DIM_COIL'
    OPERATION = 'Coil Sampling'
    MODES = ('synthesize', 'random', 'deterministic')

    # Drawing runs natively anywhere; synthesis narrows this per instance,
    # because a NIfTI list holds no coil axis to write into.
    SUPPORTED_BACKENDS = tuple(Backend)

    def __init__(self, mode: str = 'random', n_coils=None, source=None, seed=None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")

        if mode == 'synthesize':
            BaseModule.__init__(self)
            self.mode = mode
            # Synthesis writes a dimension that was not there, and only the
            # module that added it knows what it is.
            self.ADDS_DIM_TAGS = (self.DIM_TAG,)
            self.SUPPORTED_BACKENDS = tuple(b for b in Backend
                                            if b is not Backend.NIFTI_LIST)
            # The source knows how many elements it has, so None leaves it to it.
            self.n_coils = None if n_coils is None else int(n_coils)
        else:
            super().__init__(mode=mode, count=n_coils, seed=seed)
            self.n_coils = self.count

        self.source = source or Birdcage()

    #***************#
    #   synthesis   #
    #***************#
    def maps_for(self, matrix) -> np.ndarray:
        """
        Sensitivity maps over a spatial matrix, from whichever source was given.

        Args:
            matrix: Spatial grid "(X, Y, Z)".

        Returns:
            Maps "(X, Y, Z, C)".
        """
        return self.source.maps(matrix, self.n_coils, self.rng.numpy_rng())

    def process_tensor(self, data_array, water_array=None,
                       backend: Backend = Backend.NUMPY, **kwargs):
        """
        Synthesize a receive array, or draw from one.

        Args:
            data_array: "(batch, X, Y, Z, T)" to synthesize onto, or a batch
                carrying DIM_COIL to draw from.
            water_array: Passed through unchanged - a water reference is a
                separate acquisition and gets its own coils if it needs them.
            backend: Backend enum (unused; kept for the BaseModule signature).
            **kwargs: Absorbs what BaseModule injects, including "dim_tags".

        Returns:
            "(data, water_unchanged)".
        """
        if self.mode != 'synthesize':
            return super().process_tensor(data_array, water_array, backend, **kwargs)

        shape = tuple(int(n) for n in ops.shape(data_array))
        if len(shape) != 5:
            raise ValueError(
                f"CoilSampler[synthesize] expects (batch, X, Y, Z, T), got {shape}. "
                f"Data that already carries a coil axis has a receive array."
            )

        maps = self.maps_for(shape[1:4])
        n_coils = maps.shape[-1]

        # Broadcast the maps across batch and spectral points, and open a coil
        # axis on the data for them to multiply into.
        maps = ops.match_backend(maps.astype(np.complex64), data_array)
        maps = ops.reshape(maps, (1,) + shape[1:4] + (1, n_coils))

        return ops.reshape(data_array, shape + (1,)) * maps, water_array
