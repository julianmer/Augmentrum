####################################################################################################
#                                     mrsi_challenge.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-29                                                                              #
#                                                                                                  #
# Purpose: Loads the MRSI Challenge dataset (FID-MRSI, 64x64x32x384 at 3T) into Augmentrum,        #
#          with the challenge's own test subjects pinned as held-out splits.                       #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from augmentrum.core.augmentrum import Augmentrum


__all__ = ['MRSIChallengeDataModule', 'MRSIChallengeData']


#**************************************************************************************************#
#                                  Class MRSIChallengeDataModule                                   #
#**************************************************************************************************#
#                                                                                                  #
# Loader for the MRSI Challenge dataset.                                                           #
#                                                                                                  #
#**************************************************************************************************#
class MRSIChallengeDataModule:
    """
    Loader for the MRSI Challenge dataset.

    The data is simulated FID-MRSI on a 64 x 64 x 32 grid with 384 spectral points,
    at 3 T. Each subject ships several signal components separately, which is what
    makes it useful here: you can train on the *clean* metabolite signal and let
    Augmentrum supply the degradation, instead of inheriting whatever corruption
    the dataset happens to contain.

    Signal components
    -----------------
    "xtMeta"       metabolites only, noiseless — the "signal='clean'" default
    "xtNuisance"   residual water + lipid, summed (not separable)
    "xtAll"        the composite: metabolites + macromolecules + baseline
                     + nuisance + noise

    **Macromolecules are not inside** "xtMeta". The release keeps them in a
    separate "xtMM" array, and that array ships only with the *test* ground-truth
    files — the 24 training subjects do not carry it at all. For those, the only
    handle on the macromolecular signal is "xtAll - xtMeta - xtNuisance", which
    is MM *plus* baseline *plus* noise and cannot be separated further.

    So there is no "metabolites + macromolecules, nothing else" option for
    training. The two usable positions are:

    * "signal='clean'" — metabolites alone, noiseless. Nothing to unpick, and
      the macromolecular baseline and noise come from Augmentrum instead, where
      they are parameterized and reproducible.
    * "signal='nuisance_free'" ("xtAll - xtNuisance") — metabolites,
      macromolecules, baseline and the challenge's own noise. This is the input
      the organizers recommend for the quantification-only sub-challenge, but
      anything Augmentrum adds then stacks on top of noise that is already there.

    Splits
    ------
    Training subjects are "Sub1..Sub24". The two test sets are the challenge's
    own held-out subjects and are pinned, never sampled:

    ==============  ======================  ==============================
    split           subjects                task
    ==============  ======================  ==============================
    "test_track1" "TestSub1..TestSub5"  nuisance removal + quantification
    "test_track2" "TestSub10..TestSub12" quantification only (no nuisance)
    ==============  ======================  ==============================

    For the test subjects the clean components live in the withheld ground-truth
    files, which this loader resolves automatically when they are present.

    Layout on disk
    --------------
    "<data_dir>/" must contain "contest_data/", "testing_data/",
    "testing_data_2/" and, for anything other than "signal='composite'" on the
    test sets, "testing_data_ground_truth/" and "testIng_data_2_ground_truth/"
    (the capital I in that last name is the organizers', not a typo here).

    Caching
    -------
    Every ".mat" is a 2.2 GB MATLAB v7.3 (HDF5) file. The requested component is
    extracted once and cached as an **uncompressed** NIfTI-MRS file, which
    "read_FID" then memory-maps: 32 full volumes cost ~0 RAM until a batch
    actually touches them. Loading them eagerly instead would be ~13 GB.

    Examples
    --------
    >>> mod = MRSIChallengeDataModule('data/MRSI_Challenge')
    >>> data, names, aux = mod.load('train', n_subjects=4)
    >>> data[0].shape
    (64, 64, 32, 384)
    """

    #: The one quantity no released file records. Every NIfTI carries a 10000 mm
    #: placeholder in "pixdim" and an identity-scaled sform, and the ".mat"
    #: has no voxel size at all, so this is derived from the FOV quoted in the
    #: dataset description (179.2 x 224.0 x 128.0 mm over a 64 x 64 x 32 matrix).
    VOXEL_MM = (2.8, 3.5, 4.0)

    #: Fallbacks, used only when a file does not record the value. The ".mat"
    #: stores "hzpppm" and "ppmoff" exactly and its "t" vector gives both
    #: dwell time and echo time, so on that path nothing here is consulted. The
    #: NIfTI path needs them: those headers round the center frequency to 127.73
    #: and omit the ppm offset entirely.
    DEFAULT_SPECTROMETER_FREQUENCY_MHZ = 127.732434     # 3 T, 1H
    DEFAULT_PPM_OFFSET = 4.65                           # ppm at zero offset
    DEFAULT_DWELL_TIME_S = 0.83e-3                      # -> 1204.8 Hz bandwidth
    DEFAULT_ECHO_TIME_S = 1.66e-3
    DEFAULT_N_POINTS = 384

    TRAIN_SUBJECTS = tuple(f"Sub{i}" for i in range(1, 25))
    TRACK1_SUBJECTS = tuple(f"TestSub{i}" for i in range(1, 6))
    TRACK2_SUBJECTS = ("TestSub10", "TestSub11", "TestSub12")

    #: signal name -> the .mat variables it is built from. More than one entry
    #: means the first minus the rest.
    SIGNALS = {
        'clean':         ('xtMeta',),                  # metabolites only, noiseless
        'metabolites':   ('xtMeta',),
        'nuisance_free': ('xtAll', 'xtNuisance'),      # metab + MM + baseline + noise
        'composite':     ('xtAll',),                   # everything, as released
        'nuisance':      ('xtNuisance',),              # residual water + lipid
    }

    #: signal name -> the NIfTI-MRS file suffixes it is built from. Training
    #: subjects ship the components as NIfTI as well as inside the .mat, and
    #: reading those is far cheaper than pulling a variable out of a 2.2 GB
    #: HDF5 file. Test subjects ship only the composite, so anything else there
    #: still has to come from the ground-truth .mat.
    NIFTI_SIGNALS = {
        'clean':         ('mrs_fids_metabolites',),
        'metabolites':   ('mrs_fids_metabolites',),
        'nuisance_free': ('mrs_fids_si_data', 'mrs_fids_nuisance'),   # difference
        'composite':     ('mrs_fids_si_data',),
        'nuisance':      ('mrs_fids_nuisance',),
    }

    SPLITS = ('train', 'test_track1', 'test_track2')

    def __init__(self,
                 data_dir: str,
                 signal: str = 'clean',
                 source: str = 'mat',
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True,
                 dtype: Any = np.complex64):
        """
        data_dir: root of the challenge release.
        signal: which component to load — see "SIGNALS".
        source: where to read the spectral data from — "'mat'" (default),
                "'nifti'", or "'auto'" (NIfTI where shipped, else .mat).

                **The .mat is the default because the two sources do not agree.**
                The release ships each component twice, and for training subjects
                the NIfTI files carry a "_v2" suffix and are ~6 weeks newer than
                the ".mat". Their composites match closely (complex correlation
                0.98 on a test voxel), but the *metabolite* components do not
                (0.19): same peaks in the magnitude spectrum, different phase, and
                a ~5% amplitude difference. Only the ".mat" is internally
                self-consistent — every component comes from one generation run,
                so "xtAll - xtNuisance" and "xtMeta" refer to the same signal.
                Mixing sources would silently break that.

                The NIfTI files are also stored **conjugated** relative to the
                ".mat" (this loader undoes that, so both give the same
                convention), and they carry no usable geometry: "pixdim" is a
                10000 mm placeholder and "SpectrometerFrequency" is rounded to
                127.73, losing four decimals. Geometry comes from the class
                constants either way.

                Test subjects ship only the composite as NIfTI, so anything else
                there comes from the withheld ground-truth ".mat" regardless.
        cache_dir: where extracted volumes are cached. Defaults to
                "<data_dir>/_augmentrum_cache".
        use_cache: set False to always re-read the source files.
        dtype: complex dtype for the cached volumes. complex64 halves both the
                cache size and the per-batch memory at no meaningful precision
                cost for this data.
        """
        if signal not in self.SIGNALS:
            raise ValueError(
                f"signal must be one of {sorted(self.SIGNALS)}, got {signal!r}"
            )
        if source not in ('auto', 'nifti', 'mat'):
            raise ValueError(f"source must be 'auto', 'nifti' or 'mat', got {source!r}")
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.signal = signal
        self.source = source
        self.use_cache = use_cache
        self.dtype = dtype
        self.cache_dir = cache_dir or os.path.join(self.data_dir, '_augmentrum_cache')

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"MRSI Challenge data directory not found: {self.data_dir}\n"
                "Point --data-dir at the release root (the folder holding "
                "contest_data/, testing_data/ and testing_data_2/)."
            )

    #*********************#
    #   path resolution   #
    #*********************#

    def subjects(self, split: str) -> Tuple[str, ...]:
        """Subject IDs belonging to *split*."""
        if split == 'train':
            return self.TRAIN_SUBJECTS
        if split == 'test_track1':
            return self.TRACK1_SUBJECTS
        if split == 'test_track2':
            return self.TRACK2_SUBJECTS
        raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

    def mat_path(self, subject: str, need_truth: bool) -> str:
        """
        Locate the .mat file holding *subject*.

        Test subjects appear twice in the release: the participant file, which has
        only the composite "xtAll", and the withheld ground-truth file, which has
        the separated components. *need_truth* selects between them.
        """
        if subject in self.TRAIN_SUBJECTS:
            return os.path.join(self.data_dir, 'contest_data', subject, f'{subject}_all.mat')

        if subject in self.TRACK1_SUBJECTS:
            participant = os.path.join(self.data_dir, 'testing_data', subject, f'{subject}.mat')
            truth = os.path.join(self.data_dir, 'testing_data_ground_truth',
                                 f'{subject}_all_truth.mat')
        elif subject in self.TRACK2_SUBJECTS:
            participant = os.path.join(self.data_dir, 'testing_data_2', subject, f'{subject}.mat')
            # The organizers' directory name really does capitalize the I.
            truth = os.path.join(self.data_dir, 'testIng_data_2_ground_truth',
                                 f'{subject}_all_truth.mat')
        else:
            raise ValueError(f"Unknown subject {subject!r}")

        if not need_truth:
            return participant
        if os.path.isfile(truth):
            return truth
        raise FileNotFoundError(
            f"signal={self.signal!r} needs the separated components, which for "
            f"{subject} live in the withheld ground-truth file:\n  {truth}\n"
            f"That file is missing. Use signal='composite' to load the participant "
            f"file ({participant}) instead."
        )

    #******************#
    #   .mat reading   #
    #******************#

    @staticmethod
    def _to_complex(dataset) -> np.ndarray:
        """
        Read an HDF5 dataset that MATLAB wrote as a complex array.

        MATLAB v7.3 stores complex data as a compound dtype with 'real' and 'imag'
        fields; h5py surfaces that verbatim rather than as a complex array.
        """
        arr = dataset[()] if hasattr(dataset, 'shape') else dataset
        if arr.dtype.names and 'real' in arr.dtype.names and 'imag' in arr.dtype.names:
            return arr['real'] + 1j * arr['imag']
        return arr

    def read_acquisition(self, subject: str) -> Dict[str, float]:
        """
        Read the acquisition parameters for *subject* from the data itself.

        The ".mat" records everything except voxel size: "hzpppm" is the exact
        center frequency, "ppmoff" the chemical-shift reference, and the "t"
        vector gives both the dwell time (its spacing) and the echo time (its
        first sample — acquisition starts at TE, not at zero). Reading them beats
        hard-coding, which is how a loader ends up silently describing a different
        acquisition than the one it loaded.

        Falls back to the "DEFAULT_*" constants only for values a given source
        genuinely does not carry.

        Returns a dict with "spectrometer_frequency_mhz", "ppm_offset",
        "dwell_time_s", "echo_time_s" and "n_points".
        """
        acq = {
            'spectrometer_frequency_mhz': self.DEFAULT_SPECTROMETER_FREQUENCY_MHZ,
            'ppm_offset':                 self.DEFAULT_PPM_OFFSET,
            'dwell_time_s':               self.DEFAULT_DWELL_TIME_S,
            'echo_time_s':                self.DEFAULT_ECHO_TIME_S,
            'n_points':                   self.DEFAULT_N_POINTS,
        }

        try:
            import h5py
            path = self.mat_path(subject, need_truth=False)
            if not os.path.isfile(path):
                path = self.mat_path(subject, need_truth=True)
            with h5py.File(path, 'r') as f:
                if 'hzpppm' in f:
                    acq['spectrometer_frequency_mhz'] = float(np.asarray(f['hzpppm'][()]).ravel()[0])
                if 'ppmoff' in f:
                    acq['ppm_offset'] = float(np.asarray(f['ppmoff'][()]).ravel()[0])
                if 't' in f:
                    t = np.asarray(f['t'][()]).ravel()
                    if t.size > 1:
                        acq['dwell_time_s'] = float(t[1] - t[0])
                        acq['echo_time_s'] = float(t[0])
                        acq['n_points'] = int(t.size)
        except Exception:
            # A missing or unreadable .mat is not fatal — the NIfTI path still
            # works, it just leans on the documented defaults.
            pass

        return acq

    @property
    def acquisition(self) -> Dict[str, float]:
        """Acquisition parameters, read once from the first available subject."""
        if getattr(self, '_acquisition', None) is None:
            self._acquisition = self.read_acquisition(self.TRAIN_SUBJECTS[0])
        return self._acquisition

    @classmethod
    def _to_nifti_order(cls, arr: np.ndarray) -> np.ndarray:
        """
        Reverse MATLAB's column-major axis order.

        A 64x64x32x384 MATLAB array reaches h5py as (384, 32, 64, 64) = (T, Z, Y, X);
        NIfTI-MRS wants (X, Y, Z, T).
        """
        return np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))

    def nifti_paths(self, subject: str) -> Optional[List[str]]:
        """
        Paths to the NIfTI-MRS files making up the configured signal, or None if
        this subject does not ship them (every test subject except the composite).

        Training subjects carry a "_v2" suffix that the test subjects lack.
        """
        suffixes = self.NIFTI_SIGNALS.get(self.signal)
        if suffixes is None:
            return None

        if subject in self.TRAIN_SUBJECTS:
            folder, tag = os.path.join(self.data_dir, 'contest_data', subject), '_v2'
        elif subject in self.TRACK1_SUBJECTS:
            folder, tag = os.path.join(self.data_dir, 'testing_data', subject), ''
        elif subject in self.TRACK2_SUBJECTS:
            folder, tag = os.path.join(self.data_dir, 'testing_data_2', subject), ''
        else:
            raise ValueError(f"Unknown subject {subject!r}")

        paths = [os.path.join(folder, f'{subject}_{s}{tag}.nii.gz') for s in suffixes]
        return paths if all(os.path.isfile(p) for p in paths) else None

    def read_component_nifti(self, subject: str) -> np.ndarray:
        """Read the configured signal component from the shipped NIfTI-MRS files."""
        import nibabel as nib

        paths = self.nifti_paths(subject)
        if paths is None:
            raise FileNotFoundError(
                f"{subject} does not ship signal={self.signal!r} as NIfTI-MRS."
            )
        arr = np.asarray(nib.load(paths[0]).dataobj).astype(self.dtype)
        for extra in paths[1:]:
            arr = arr - np.asarray(nib.load(extra).dataobj).astype(self.dtype)
        return np.ascontiguousarray(arr)

    def read_component(self, subject: str) -> np.ndarray:
        """
        Read the configured signal component for *subject* as (X, Y, Z, T).

        Honours "source": NIfTI when available and permitted, otherwise the .mat.
        """
        if self.source in ('auto', 'nifti'):
            paths = self.nifti_paths(subject)
            if paths is not None:
                return self.read_component_nifti(subject)
            if self.source == 'nifti':
                raise FileNotFoundError(
                    f"source='nifti' but {subject} does not ship signal="
                    f"{self.signal!r} as NIfTI-MRS. Test subjects ship only the "
                    f"composite; use source='auto' to fall back to the .mat."
                )

        import h5py

        variables = self.SIGNALS[self.signal]
        need_truth = variables != ('xtAll',)
        path = self.mat_path(subject, need_truth=need_truth)

        with h5py.File(path, 'r') as f:
            missing = [v for v in variables if v not in f]
            if missing:
                raise KeyError(
                    f"{os.path.basename(path)} has no {missing}; it contains "
                    f"{sorted(k for k in f.keys() if not k.startswith('#'))}."
                )
            arr = self._to_complex(f[variables[0]])
            for extra in variables[1:]:
                arr = arr - self._to_complex(f[extra])

        return np.ascontiguousarray(self._to_nifti_order(arr).astype(self.dtype))

    def read_aux(self, subject: str) -> Dict[str, np.ndarray]:
        """
        Read the per-subject auxiliary maps: brain mask, B0 map, anatomical
        reference and — where available — the ground-truth metabolite amplitudes.

        All are returned in (X, Y, Z) order to match the spectral volumes.
        "metaMap" gains a trailing metabolite axis: (X, Y, Z, n_metabolites).
        """
        import h5py

        path = self.mat_path(subject, need_truth=self.SIGNALS[self.signal] != ('xtAll',))
        aux: Dict[str, np.ndarray] = {}
        with h5py.File(path, 'r') as f:
            for key in ('brainMask', 'B0map', 'Iref', 'metaMap'):
                if key in f:
                    aux[key] = self._to_nifti_order(np.asarray(f[key][()]))
            for key in ('hzpppm', 'ppmoff'):
                if key in f:
                    aux[key] = float(np.asarray(f[key][()]).ravel()[0])
            if 't' in f:
                aux['t'] = np.asarray(f['t'][()]).ravel()
        return aux

    #****************************************#
    #   nifti-mrs construction and caching   #
    #****************************************#

    def _cache_path(self, subject: str) -> str:
        return os.path.join(self.cache_dir, f'{subject}_{self.signal}.nii')

    def to_nifti(self, fids: np.ndarray, acquisition: Optional[Dict[str, float]] = None):
        """
        Wrap an (X, Y, Z, T) array as a NIfTI-MRS object with correct geometry.

        Unlike the files this data came from, the result carries a real voxel
        size and the full-precision center frequency, so downstream consumers can
        read geometry off the object instead of being told it separately.

        "no_conj=False" stores the FIDs exactly as released, which is the
        orientation FSL-MRS expects. Note the flag reads backwards: "no_conj=True"
        *applies* a conjugation. Both released sources already store FIDs such
        that "fft" gives a correctly ordered spectrum, and "FIDToSpec" — the
        transform behind "MRS.get_spec()", "NIfTI_MRS_Plus.plot" and the
        spectral augmentations — uses "fft". Conjugating here would mirror the
        axis and put NAA at 7.3 ppm instead of 2.008 for every one of them.
        """
        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

        acq = acquisition or self.acquisition
        affine = np.diag(list(self.VOXEL_MM) + [1.0])
        nifti = gen_nifti_mrs(
            data=fids,
            dwelltime=acq['dwell_time_s'],
            spec_freq=acq['spectrometer_frequency_mhz'],
            nucleus='1H',
            dim_tags=[None, None, None],
            no_conj=False,
            affine=affine,
        )
        try:
            nifti.add_hdr_field('EchoTime', acq['echo_time_s'])
        except Exception:
            pass       # header extension is optional metadata, not worth failing over
        return nifti

    def load_subject(self, subject: str):
        """
        Return one subject as a NIfTI-MRS object, using the cache when possible.

        The cache is written uncompressed so "read_FID" can memory-map it; a
        gzipped file would have to be inflated into RAM in full.
        """
        from fsl_mrs.utils import mrs_io

        cache = self._cache_path(subject)
        if self.use_cache and os.path.isfile(cache):
            return mrs_io.read_FID(cache)

        nifti = self.to_nifti(self.read_component(subject))

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            nifti.save(cache)
            return mrs_io.read_FID(cache)        # reopen memory-mapped
        return nifti

    def load(self, split: str, n_subjects: Optional[int] = None,
             with_aux: bool = False) -> Tuple[List, List[str], List[Dict]]:
        """
        Load a split.

        Args:
            split: 'train', 'test_track1' or 'test_track2'.
            n_subjects: keep only the first N subjects. Useful for a quick run —
                a full training subject is ~400 MB of cache.
            with_aux: also read the auxiliary maps (brain mask, B0, metaMap).

        Returns:
            (nifti_list, subject_names, aux_list). "aux_list" is a list of empty
            dicts when *with_aux* is False.
        """
        names = list(self.subjects(split))
        if n_subjects is not None:
            names = names[:int(n_subjects)]

        data, aux = [], []
        for name in names:
            data.append(self.load_subject(name))
            aux.append(self.read_aux(name) if with_aux else {})
        return data, names, aux

    #***************************#
    #   spectral axis helpers   #
    #***************************#

    def ppm_axis(self, n_points: Optional[int] = None,
                 dwell_time: Optional[float] = None,
                 hzpppm: Optional[float] = None,
                 ppm_offset: Optional[float] = None) -> np.ndarray:
        """
        Chemical-shift axis for this dataset, in ppm.

        Uses the release convention "ppm = f / hzpppm + ppmoff", with both
        constants read from the data by "read_acquisition" unless overridden
        here. Note the shipped example scripts use 123.23 and 4.7, which belong to
        an older example dataset and put NAA visibly off its 2.008 ppm position on
        the release data.
        """
        acq = self.acquisition
        n = int(n_points or acq['n_points'])
        dt = float(dwell_time or acq['dwell_time_s'])
        hz = float(hzpppm or acq['spectrometer_frequency_mhz'])
        off = acq['ppm_offset'] if ppm_offset is None else float(ppm_offset)
        freq = np.linspace(-1.0 / (2.0 * dt), 1.0 / (2.0 * dt), n)
        return freq / hz + off

    def to_spectrum(self, fid: np.ndarray, axis: int = -1,
                    line_broadening_hz: float = 0.0,
                    dwell_time: Optional[float] = None,
                    echo_time: Optional[float] = None) -> np.ndarray:
        """
        Transform FIDs to spectra on the axis returned by "ppm_axis".

        Uses "fft", matching FSL-MRS's "FIDToSpec" so that this helper and
        "MRS.get_spec()" agree on the same data. Swapping in "ifft" mirrors
        the axis and lands NAA at 7.3 ppm instead of 2.008.

        Also applies the first-order phase for the echo time, without which the
        spectra come out badly phased: sampling starts at TE, not at t=0.
        """
        acq = self.acquisition
        dt = float(dwell_time or acq['dwell_time_s'])
        t0 = float(acq['echo_time_s'] if echo_time is None else echo_time)
        n = fid.shape[axis]

        t = t0 + np.arange(n) * dt
        freq = np.linspace(-1.0 / (2.0 * dt), 1.0 / (2.0 * dt), n)

        shape = [1] * fid.ndim
        shape[axis] = n
        weighting = np.exp(-np.pi * float(line_broadening_hz) * (t - t0)).reshape(shape)
        first_order = np.exp(-1j * 2.0 * np.pi * freq * t0).reshape(shape)

        spec = np.fft.fftshift(np.fft.fft(fid * weighting, axis=axis), axes=axis)
        return spec * first_order


def MRSIChallengeData(data_dir: str,
                      signal: str = 'clean',
                      n_train: Optional[int] = None,
                      n_val: int = 5,
                      batch_size: int = 2,
                      seed: int = 42,
                      baseline: bool = False,
                      pipelines: Optional[Dict[str, Any]] = None,
                      modes: Optional[Dict[str, str]] = None,
                      backend: str = 'pytorch',
                      volatile: bool = True,
                      cache_dir: Optional[str] = None,
                      use_cache: bool = True,
                      with_aux: bool = False,
                      **kwargs) -> Augmentrum:
    """
    Load the MRSI Challenge into an "~augmentrum.core.augmentrum.Augmentrum".

    Train and validation are carved out of the 24 contest subjects; both test sets
    are the challenge's own held-out subjects and are pinned by index rather than
    sampled, so they can never leak into training.

    Args:
        data_dir: root of the challenge release.
        signal: which component to load (see "MRSIChallengeDataModule").
                Defaults to the clean, noiseless metabolite signal.
        n_train: total contest subjects to use, train + val. None uses all 24.
        n_val: how many of those are held out for validation.
        batch_size: volumes per batch. A full volume is ~400 MB as complex64, so
                2-4 is the practical range; 16 would need ~10 GB of working set.
        baseline: include the spectral baseline augmentation. Off by default
                because it is per-voxel and costs ~400 s per volume on CPU,
                against ~25 s for the rest of the pipeline combined. Turn it on
                when you want a synthetic macromolecular baseline and can afford
                it (or are precomputing an augmented cache rather than
                augmenting inside the training loop).
        pipelines, modes: per-split overrides. The default trains with spatial
                augmentation, k-space undersampling and noise, validates with
                undersampling and noise at fixed parameters, and leaves both test
                sets untouched.
        with_aux: attach the per-subject brain mask, B0 map and metabolite maps to
                the returned object as ".aux" (a dict keyed by split).
        **kwargs: module parameters forwarded to Augmentrum, e.g.
                "acceleration_factor=(2.0, 6.0)" or "sigma=1e-3".

    Returns:
        Augmentrum with splits 'train', 'val', 'test_track1' and 'test_track2'.

    Examples:
        >>> aug = MRSIChallengeData('data/MRSI_Challenge', n_train=8,
        ...                         acceleration_factor=(2.0, 6.0), sigma=(0.6e-3, 1.4e-3))
        >>> batch, _ = next(aug.train_dataloader())
        >>> batch.shape
        torch.Size([2, 64, 64, 32, 384])
    """
    module = MRSIChallengeDataModule(data_dir, signal=signal,
                                     cache_dir=cache_dir, use_cache=use_cache)

    contest, contest_names, contest_aux = module.load('train', n_subjects=n_train,
                                                      with_aux=with_aux)
    track1, track1_names, track1_aux = module.load('test_track1', with_aux=with_aux)
    track2, track2_names, track2_aux = module.load('test_track2', with_aux=with_aux)

    if n_val >= len(contest):
        raise ValueError(
            f"n_val={n_val} leaves no training subjects — only {len(contest)} "
            f"contest subjects were loaded."
        )

    # Validation is the tail of the contest list, so it stays stable as n_train grows.
    n_train_actual = len(contest) - n_val
    all_data = list(contest) + list(track1) + list(track2)
    n_c, n_1 = len(contest), len(track1)

    split_indices = {
        'train':       list(range(0, n_train_actual)),
        'val':         list(range(n_train_actual, n_c)),
        'test_track1': list(range(n_c, n_c + n_1)),
        'test_track2': list(range(n_c + n_1, len(all_data))),
    }

    if pipelines is None:
        # signal='clean' is metabolites alone — no macromolecules, no baseline, no
        # noise (see MRSIChallengeDataModule for why MM cannot be recovered
        # separately for the training subjects). Augmentrum supplies the missing
        # realism, which is the point: it is parameterized and reproducible,
        # whereas whatever the release happens to contain is fixed.
        #
        # Measured on one 64x64x32x384 volume, CPU: spatial 4.7 s, undersampling
        # 10.8 s, noise 9.0 s — and baseline ~400 s, because it draws an
        # independent random walk per voxel and there are 131072 of them. Include
        # it only if you want that, and see `baseline` in the docstring below.
        pipelines = {
            'train':       ['spatial', 'undersampling', 'noise'],
            'val':         ['undersampling', 'noise'],
            'test_track1': [],
            'test_track2': [],
        }
        if baseline:
            pipelines['train'].insert(2, 'baseline')
            pipelines['val'].insert(1, 'baseline')
    if modes is None:
        modes = {
            'train':       'on-the-fly',
            'val':         'fixed',
            'test_track1': 'fixed',
            'test_track2': 'fixed',
        }

    # Defaults suited to this data. Spatial augmentation needs the real voxel size
    # to rotate physically, and noise needs an absolute sigma because a per-voxel
    # statistic would leave the background noiseless.
    kwargs.setdefault('pixdim', MRSIChallengeDataModule.VOXEL_MM)
    kwargs.setdefault('allow_rot90', False)      # 179.2 x 224.0 mm is not square
    kwargs.setdefault('global_scale', True)
    kwargs.setdefault('sigma', 1.0e-3)           # the challenge's own training sigma

    aug = Augmentrum(
        data=all_data,
        split_indices=split_indices,
        pipelines=pipelines,
        modes=modes,
        batch_size=batch_size,
        backend=backend,
        volatile=volatile,
        seed=seed,
        **kwargs,
    )

    aug.subject_names = {
        'train':       contest_names[:n_train_actual],
        'val':         contest_names[n_train_actual:],
        'test_track1': track1_names,
        'test_track2': track2_names,
    }
    if with_aux:
        aug.aux = {
            'train':       contest_aux[:n_train_actual],
            'val':         contest_aux[n_train_actual:],
            'test_track1': track1_aux,
            'test_track2': track2_aux,
        }
    aug.data_module = module
    return aug
