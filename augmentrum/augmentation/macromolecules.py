####################################################################################################
#                                      macromolecules.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-11                                                                              #
#                                                                                                  #
# Purpose: Adds a macromolecular baseline to MRS data, from a parametrized, semi-parametrized,     #
#          measured, or user-supplied MM source.                                                   #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import re
import struct
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from augmentrum.core.base_module import BaseModule
from augmentrum.processing.domain import Domain
from nifti_mrs_plus import Backend
from nifti_mrs_plus import ops
from nifti_mrs_plus.ops import match_backend


#: Approximate consensus macromolecular components as (ppm, FWHM_ppm, rel_amp).
#: Positions follow the experts' consensus (Cudalbu et al., NMR Biomed 2021,
#: doi:10.1002/nbm.4393; M0.94 ... M4.03); widths and relative amplitudes are
#: representative healthy-brain defaults meant to be jittered, not fitted.
MM_CONSENSUS: Tuple[Tuple[float, float, float], ...] = (
    (0.92, 0.18, 1.00),
    (1.21, 0.16, 0.35),
    (1.39, 0.14, 0.40),
    (1.67, 0.15, 0.15),
    (2.04, 0.17, 0.75),
    (2.26, 0.18, 0.50),
    (2.56, 0.14, 0.10),
    (2.70, 0.14, 0.08),
    (2.99, 0.16, 0.35),
    (3.21, 0.16, 0.25),
    (3.62, 0.14, 0.15),
    (3.75, 0.16, 0.30),
    (3.86, 0.14, 0.15),
    (4.03, 0.15, 0.12),
)


#**************************************************************************************************#
#                                         Class MMSource                                           #
#**************************************************************************************************#
#                                                                                                  #
# Where a macromolecular spectrum comes from.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class MMSource(ABC):
    """
    Where a macromolecular spectrum comes from.

    A source produces a complex MM profile on a requested ppm axis, normalized
    to unit maximum of its real part, so the module can scale it against the
    data. Randomizable sources draw from the generator they are handed, which
    keeps every draw reproducible from the module's seed.
    """

    @abstractmethod
    def profile(self, ppm_axis: np.ndarray, rng: np.random.Generator,
                sf_mhz: Optional[float] = None) -> np.ndarray:
        """The unit-normalized complex MM spectrum on *ppm_axis*."""

    @staticmethod
    def _normalize(spectrum: np.ndarray) -> np.ndarray:
        """Scale a profile so its real part peaks at 1."""
        peak = np.max(np.abs(np.real(spectrum)))
        return spectrum / peak if peak > 0 else spectrum

    @staticmethod
    def _regrid(spectrum: np.ndarray, ppm_from: np.ndarray,
                ppm_to: np.ndarray) -> np.ndarray:
        """
        Interpolate a profile from its own ppm axis onto the data's.

        Outside the source's coverage the MM signal is genuinely absent, so the
        profile goes to zero there rather than being extrapolated.
        """
        order = np.argsort(ppm_from)
        real = np.interp(ppm_to, ppm_from[order], np.real(spectrum)[order],
                         left=0.0, right=0.0)
        imag = np.interp(ppm_to, ppm_from[order], np.imag(spectrum)[order],
                         left=0.0, right=0.0)
        return real + 1j * imag

    @staticmethod
    def _fid_to_spectrum(fid: np.ndarray, sw_hz: float, sf_mhz: float,
                         ref_ppm: float = 4.7) -> Tuple[np.ndarray, np.ndarray]:
        """A measured FID as (spectrum, ppm_axis), MRS convention."""
        fid = np.asarray(fid).squeeze()
        if fid.ndim > 1:                       # average any dynamics
            fid = fid.reshape(fid.shape[0], -1).mean(axis=1)
        spectrum = np.fft.fftshift(np.fft.ifft(fid))
        freq_hz = np.fft.fftshift(np.fft.fftfreq(fid.shape[0], d=1.0 / sw_hz))
        return spectrum, ref_ppm - freq_hz / sf_mhz


#**************************************************************************************************#
#                                       Class Parametrized                                         #
#**************************************************************************************************#
#                                                                                                  #
# Sum of Gaussian components at the consensus MM positions.                                        #
#                                                                                                  #
#**************************************************************************************************#
class Parametrized(MMSource):
    """
    Sum of Gaussian components at the consensus MM positions.

    Each component is (ppm, FWHM_ppm, rel_amp); the defaults are the
    "MM_CONSENSUS" table. The jitters are fractional (amplitude, width) or
    absolute in ppm (position) half-ranges drawn uniformly per call, which is
    what turns a fixed template into an augmentation.
    """

    def __init__(self, components: Tuple = MM_CONSENSUS,
                 amp_jitter: float = 0.0,
                 ppm_jitter: float = 0.0,
                 fwhm_jitter: float = 0.0):
        self.components = tuple(components)
        self.amp_jitter = float(amp_jitter)
        self.ppm_jitter = float(ppm_jitter)
        self.fwhm_jitter = float(fwhm_jitter)

    def profile(self, ppm_axis, rng, sf_mhz=None):
        ppm = np.asarray(ppm_axis, float)
        spectrum = np.zeros_like(ppm)

        for center, fwhm, amp in self.components:
            if self.ppm_jitter:
                center = center + rng.uniform(-self.ppm_jitter, self.ppm_jitter)
            if self.fwhm_jitter:
                fwhm = fwhm * (1.0 + rng.uniform(-self.fwhm_jitter, self.fwhm_jitter))
            if self.amp_jitter:
                amp = amp * (1.0 + rng.uniform(-self.amp_jitter, self.amp_jitter))

            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            spectrum += amp * np.exp(-0.5 * ((ppm - center) / sigma) ** 2)

        return self._normalize(spectrum.astype(complex))


#**************************************************************************************************#
#                                          Class Supplied                                          #
#**************************************************************************************************#
#                                                                                                  #
# A user-provided MM spectrum: array + ppm axis, or a file on disk.                                #
#                                                                                                  #
#**************************************************************************************************#
class Supplied(MMSource):
    """
    A user-provided MM spectrum: array + ppm axis, or a file on disk.

    Accepted forms:
    - "spectrum" + "ppm": a complex spectrum on its own ppm axis.
    - "path" to a ".npy" holding "[ppm, real, imag]" rows or a complex spectrum
      (then "ppm" must be given), or to a MATLAB ".mat" with an "exptDat"
      struct carrying "fid" / "sf" / "sw_h" — the layout the COWS release uses.
    """

    def __init__(self, spectrum=None, ppm=None, path: Optional[str] = None):
        if path is not None:
            spectrum, ppm = self._load(Path(path), ppm)
        if spectrum is None or ppm is None:
            raise ValueError("Supplied needs spectrum+ppm, or a readable path.")
        self._spectrum = self._normalize(np.asarray(spectrum, complex))
        self._ppm = np.asarray(ppm, float)

    @staticmethod
    def _load(path: Path, ppm):
        if path.suffix == '.npy':
            arr = np.load(path)
            if np.iscomplexobj(arr):
                if ppm is None:
                    raise ValueError(f"{path} holds a complex spectrum; pass its ppm axis.")
                return arr, ppm
            return arr[1] + 1j * arr[2], arr[0]

        if path.suffix == '.mat':
            import scipy.io as sio
            expt = sio.loadmat(str(path))['exptDat'][0, 0]
            fid = np.asarray(expt['fid']).squeeze()
            sw_hz = float(np.ravel(expt['sw_h'])[0])
            sf_mhz = float(np.ravel(expt['sf'])[0])
            return MMSource._fid_to_spectrum(fid, sw_hz, sf_mhz)

        raise ValueError(f"Unsupported MM file type: {path}")

    def profile(self, ppm_axis, rng, sf_mhz=None):
        return self._regrid(self._spectrum, self._ppm, np.asarray(ppm_axis, float))


#**************************************************************************************************#
#                                          Class Measured                                          #
#**************************************************************************************************#
#                                                                                                  #
# A measured MM spectrum from the MRSHub consensus collection, closest match first.                #
#                                                                                                  #
#**************************************************************************************************#
class Measured(MMSource):
    """
    A measured MM spectrum from the MRSHub consensus collection.

    The collection (github.com/mrshub/mm-consensus-data-collection, BSD-3) is
    downloaded once into the Augmentrum cache. Its "MM_database" holds Varian
    FID sets named "{field}T_MM_{species}_{sequence}_{site}.fid"; the source
    picks the entry whose field strength is closest to the data's (from
    "sf_mhz" at call time, or "field_t" if given), preferring the requested
    species.

    Args:
        field_t: Field strength to match. None reads it off the data.
        species: 'human' (default) or another species in the collection.
        sequence: Optional sequence filter, e.g. 'STEAM'.
    """

    ARCHIVE_URL = ("https://github.com/mrshub/mm-consensus-data-collection/"
                   "archive/refs/heads/master.tar.gz")
    GAMMA_1H = 42.577  # MHz/T

    def __init__(self, field_t: Optional[float] = None, species: str = 'human',
                 sequence: Optional[str] = None):
        self.field_t = field_t
        self.species = species
        self.sequence = sequence
        self._cache = {}

    #****************#
    #   collection   #
    #****************#
    def _database(self) -> Path:
        """The MM_database directory, downloading the collection on first use."""
        from augmentrum.utils.download import cache_root, fetch

        root = cache_root() / 'mm_consensus'
        hits = list(root.glob('*/MM_database'))
        if hits:
            return hits[0]

        root.mkdir(parents=True, exist_ok=True)
        archive = fetch(self.ARCHIVE_URL, root / 'mm-consensus.tar.gz')
        with tarfile.open(archive) as tar:
            tar.extractall(root)
        return next(root.glob('*/MM_database'))

    def _select(self, target_field: float) -> Path:
        """The .fid set closest in field strength, preferring the species asked for."""
        entries = []
        for fid_dir in sorted(self._database().glob('*.fid')):
            m = re.match(r'([\d.]+)T_MM_(\w+?)_([A-Za-z]+)_(\w+)$', fid_dir.stem)
            if not m:
                continue
            field, species, sequence = float(m.group(1)), m.group(2), m.group(3)
            if self.sequence and sequence.lower() != self.sequence.lower():
                continue
            entries.append((species != self.species, abs(field - target_field), fid_dir))

        if not entries:
            raise FileNotFoundError(
                f"No MM_database entry matches species={self.species!r}, "
                f"sequence={self.sequence!r}."
            )
        return min(entries)[2]

    #********************#
    #   varian reading   #
    #********************#
    @staticmethod
    def _read_procpar(path: Path) -> dict:
        """The scalar parameters this source needs, from a Varian procpar."""
        values, lines = {}, path.read_text(errors='ignore').splitlines()
        for i, line in enumerate(lines):
            name = line.split(' ')[0]
            if name in ('sfrq', 'sw') and i + 1 < len(lines):
                parts = lines[i + 1].split()
                if len(parts) >= 2:
                    values[name] = float(parts[1])
        return values

    @staticmethod
    def _read_fid(path: Path) -> np.ndarray:
        """The first block of a Varian fid file as a complex FID."""
        raw = path.read_bytes()
        nblocks, ntraces, npts, ebytes, tbytes, bbytes, vers, status, nbhead = \
            struct.unpack('>6lhhl', raw[:32])

        dtype = '>f4' if status & 0x8 else ('>i4' if ebytes == 4 else '>i2')
        data = np.frombuffer(raw, dtype=dtype, count=npts,
                             offset=32 + 28 * max(nbhead, 1))
        return data[0::2].astype(float) + 1j * data[1::2].astype(float)

    def profile(self, ppm_axis, rng, sf_mhz=None):
        target_field = self.field_t
        if target_field is None:
            if sf_mhz is None:
                raise ValueError("Measured needs field_t, or data with sf_mhz.")
            target_field = float(sf_mhz) / self.GAMMA_1H

        fid_dir = self._select(target_field)
        if fid_dir not in self._cache:
            params = self._read_procpar(fid_dir / 'procpar')
            fid = self._read_fid(fid_dir / 'fid')
            spectrum, ppm = self._fid_to_spectrum(fid, params['sw'], params['sfrq'])
            self._cache[fid_dir] = (self._normalize(spectrum), ppm)

        spectrum, ppm = self._cache[fid_dir]
        return self._regrid(spectrum, ppm, np.asarray(ppm_axis, float))


#**************************************************************************************************#
#                                      Class SemiParametrized                                      #
#**************************************************************************************************#
#                                                                                                  #
# A base MM lineshape with random broadening and a smooth amplitude envelope.                      #
#                                                                                                  #
#**************************************************************************************************#
class SemiParametrized(MMSource):
    """
    A base MM lineshape with random broadening and a smooth amplitude envelope.

    This is the middle ground the fitting literature uses: the overall shape is
    trusted (measured, or the parametrized template by default) while width and
    regional amplitude stay free. Per call the base profile is Gaussian-
    broadened by a draw from "broaden_ppm" and modulated by a smooth random
    envelope of relative depth "amp_mod".
    """

    def __init__(self, base: Optional[MMSource] = None,
                 broaden_ppm: Tuple[float, float] = (0.0, 0.05),
                 amp_mod: float = 0.2):
        self.base = base if base is not None else Parametrized()
        self.broaden_ppm = tuple(broaden_ppm)
        self.amp_mod = float(amp_mod)

    def profile(self, ppm_axis, rng, sf_mhz=None):
        ppm = np.asarray(ppm_axis, float)
        spectrum = self.base.profile(ppm, rng, sf_mhz=sf_mhz)

        # Broaden by convolution with a Gaussian of the drawn width. The axis
        # is uniform (it comes from an FFT grid), so the kernel is one stencil.
        width = rng.uniform(*self.broaden_ppm)
        if width > 0:
            dppm = abs(ppm[1] - ppm[0])
            sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0))) / dppm
            half = int(np.ceil(4 * sigma))
            kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
            kernel /= kernel.sum()
            spectrum = (np.convolve(np.real(spectrum), kernel, mode='same')
                        + 1j * np.convolve(np.imag(spectrum), kernel, mode='same'))

        # A slow cosine-series envelope: smooth regional amplitude freedom
        # without introducing new peaks.
        if self.amp_mod > 0:
            x = np.linspace(0.0, np.pi, ppm.size)
            envelope = np.ones_like(x)
            for k in (1, 2, 3):
                envelope += rng.uniform(-1.0, 1.0) * np.cos(k * x) / k
            envelope = 1.0 + self.amp_mod * (envelope - envelope.mean())
            spectrum = spectrum * np.clip(envelope, 0.0, None)

        return self._normalize(spectrum)


#**************************************************************************************************#
#                                       Class Macromolecules                                       #
#**************************************************************************************************#
#                                                                                                  #
# Add a macromolecular baseline to MRS data.                                                       #
#                                                                                                  #
#**************************************************************************************************#
class Macromolecules(BaseModule):
    """
    Add a macromolecular baseline to MRS data.

    The MM contribution is real signal from the imaged tissue, not an
    artifact: place this module *before* a supervised tap so both input and
    target carry it. The profile comes from an "MMSource" and is scaled
    against each FID like "ResidualWater" scales its water lobes.

    Parameters
    ----------
    mm_source : str or MMSource
        'parametrized' (consensus Gaussians), 'semi_parametrized' (base shape,
        free width/envelope), 'measured' (MRSHub consensus collection, closest
        field match), 'supplied' (your own spectrum) — or any MMSource
        instance for full control.
    mm_scale : float
        MM amplitude relative to the spectrum's real max (default 0.15).
        Pass a (min, max) range to Augmentrum to sample it per batch.
    source_params : dict
        Forwarded to the source's constructor when *mm_source* is a name.

    Examples
    --------
    >>> mm = Macromolecules()                                  # consensus template
    >>> mm = Macromolecules(mm_source='parametrized',
    ...                     source_params={'amp_jitter': 0.3, 'ppm_jitter': 0.02})
    >>> mm = Macromolecules(mm_source='measured', mm_scale=0.2)
    >>> mm = Macromolecules(mm_source=Supplied(path='sub-01_COWS7_MM.mat'))
    """

    # Tensor-only: on a NIfTI-list batch the pipeline routes to a tensor
    # backend, exactly like KspaceUndersampling.
    SUPPORTED_BACKENDS = tuple(b for b in Backend if b is not Backend.NIFTI_LIST)

    # An MM profile lives at ppm positions, which only exist in a spectrum.
    DOMAIN = Domain(spectral='frequency')

    SOURCES = {
        'parametrized': Parametrized,
        'semi_parametrized': SemiParametrized,
        'measured': Measured,
        'supplied': Supplied,
    }

    def __init__(self, mm_source='parametrized', mm_scale: float = 0.15,
                 source_params: Optional[dict] = None, seed: Optional[int] = None):
        """Initialize the module; see the class docstring for the parameters."""
        super().__init__()

        self.mm_scale = mm_scale
        if isinstance(mm_source, MMSource):
            self.source = mm_source
        elif mm_source in self.SOURCES:
            self.source = self.SOURCES[mm_source](**(source_params or {}))
        else:
            raise ValueError(
                f"mm_source must be an MMSource or one of {list(self.SOURCES)}, "
                f"got {mm_source!r}."
            )

    def process_tensor(self, data_array, water_array=None, backend=None, **kwargs):
        """
        Add the MM profile to spectra on any tensor backend.

        The profile is built once per batch in NumPy (drawing any source
        randomness from this module's seeded generator) and promoted to the
        data's backend; only the per-FID amplitude touches the data itself, so
        gradients and device placement survive.
        """
        sw_hz = kwargs.get('sw_hz')
        sf_mhz = kwargs.get('sf_mhz')
        if sw_hz is None or sf_mhz is None:
            raise ValueError("Macromolecules.process_tensor requires 'sw_hz' and "
                             "'sf_mhz' in kwargs")

        spec = data_array
        n_points = spec.shape[-1]

        freq_hz = np.fft.fftshift(np.fft.fftfreq(n_points, d=1.0 / float(sw_hz)))
        ppm = 4.7 - freq_hz / float(sf_mhz)

        unit = self.source.profile(ppm, self.rng.numpy_rng(), sf_mhz=float(sf_mhz))

        amp = self.mm_scale * ops.amax(ops.abs(ops.real(spec)), axis=-1, keepdims=True)
        mm_add = ops.cast_like(match_backend(unit, spec), spec) * ops.cast_like(amp, spec)

        return spec + mm_add, water_array
