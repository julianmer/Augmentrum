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
#          fetching subjects from Zenodo as they are needed, with the challenge's own test          #
#          subjects pinned as held-out splits.                                                     #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

    Signals
    -------
    What to load is written as a sum or difference of the release's
    components - "'meta+mm'", "'all-nuisance'", "'meta+mm+baseline'" - or as
    one of the "PRESETS" ("'clean'" is "'meta'", "'nuisance_free'" is
    "'all-nuisance'"). Any combination of these five is a valid signal:

    ==========  ==============  ===============================  =================
    component   .mat variable   what it is                       who ships it
    ==========  ==============  ===============================  =================
    "meta"      "xtMeta"        metabolites, noiseless           train, test truth
    "mm"        "xtMM"          macromolecules, noiseless        test truth
    "baseline"  "xtBaseline"    baseline, noiseless              test truth
    "nuisance"  "xtNuisance"    water + lipid, noiseless         train, track 1
    "all"       "xtAll"         everything above, plus noise     every subject
    ==========  ==============  ===============================  =================

    The last column is the only rule: a subject that lacks a component of
    the signal raises, naming what its file holds, instead of substituting
    something else. Track 2 injected no nuisance, so there "'all'" already
    is nuisance-free and "'all-nuisance'" fails.

    Every reading method takes the signal per call and the factory takes one
    per split, which is how the two positions the data offers are combined:
    train on "'clean'" and let Augmentrum add macromolecules, baseline and
    noise where they are parameterized and reproducible, then evaluate the
    test subjects against "'meta+mm'", the ground truth the release separates
    only for them. Presets and expressions that name the same thing share
    one cache file.

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

    "resolve" turns a splits spec into subject names: each entry is a count or
    an explicit sequence, "train" and "val" both draw from the contest subjects
    (train first), and a split left out is neither loaded nor fetched.

    For the test subjects the clean components live in the ground-truth files,
    which ship alongside the participant files.

    Fetching and layout on disk
    ---------------------------
    The release lives on Zenodo (doi:10.5281/zenodo.21890222) as one zip per
    subject, 3.7-4.7 GB each. "fetch" downloads and unpacks the subjects it is
    given, verifying each against its published checksum, and "load" does so
    on demand for whatever it is asked for, so a run over six subjects costs
    six downloads and not the 140 GB record. Each subject unpacks to
    "<data_dir>/<Subject>/" holding "<Subject>_all.mat", the NIfTI-MRS files
    and, for test subjects, "<Subject>_all_truth.mat".

    The data derives from WU-Minn HCP anatomies and is distributed under the
    HCP Open Access Data Use Terms; using it means citing "CITATION".

    Caching
    -------
    Every ".mat" is a 2.2 GB MATLAB v7.3 (HDF5) file. The requested component is
    extracted once and cached as an **uncompressed** NIfTI-MRS file, which
    "read_FID" then memory-maps: 32 full volumes cost ~0 RAM until a batch
    actually touches them. Loading them eagerly instead would be ~13 GB.

    Examples
    --------
    >>> mod = MRSIChallengeDataModule()
    >>> data, aux = mod.load(['Sub1', 'Sub2'])       # fetched first if absent
    >>> data[0].shape
    (64, 64, 32, 384)
    >>> truth, _ = mod.load(['TestSub1'], signal='meta+mm')
    """

    RECORD = 21890222
    DOI = '10.5281/zenodo.21890222'
    LICENSE = 'WU-Minn HCP Consortium Open Access Data Use Terms'
    CITATION = ('LaMaster J, Merkofer JP, van de Sande D, Soher B, Strasser B, Ma C. '
                'The 2024 MRSI Data Processing and Quantification Challenge Synthetic '
                'Dataset (2026). doi:10.5281/zenodo.21890222')

    #: Voxel size along (X, Y, Z) of the NIfTI grid, which the .mat path is
    #: brought onto as well. The NIfTI headers record it; the ".mat" has no
    #: voxel size at all, so it is kept here (179.2 x 224.0 x 128.0 mm over a
    #: 64 x 64 x 32 matrix, as the dataset description quotes).
    VOXEL_MM = (2.8, 3.5, 4.0)

    #: Fallbacks, used only when a file does not record the value. The ".mat"
    #: stores "hzpppm" and "ppmoff" exactly and its "t" vector gives both
    #: dwell time and echo time, so on that path nothing here is consulted. The
    #: NIfTI path needs the ppm offset, which its headers omit. The values are
    #: the release's own: the data was simulated at the Siemens nominal 3 T
    #: (~2.9 T), and with these NAA lands within one spectral bin of 2.008 ppm,
    #: where 127.732434 MHz and 4.65 ppm put it ~0.07 ppm too high.
    DEFAULT_SPECTROMETER_FREQUENCY_MHZ = 123.24
    DEFAULT_PPM_OFFSET = 4.7                            # ppm at zero offset
    DEFAULT_DWELL_TIME_S = 0.83e-3                      # -> 1204.8 Hz bandwidth
    DEFAULT_ECHO_TIME_S = 1.66e-3
    DEFAULT_N_POINTS = 384

    TRAIN_SUBJECTS = tuple(f"Sub{i}" for i in range(1, 25))
    TRACK1_SUBJECTS = tuple(f"TestSub{i}" for i in range(1, 6))
    TRACK2_SUBJECTS = ("TestSub10", "TestSub11", "TestSub12")
    ALL_SUBJECTS = TRAIN_SUBJECTS + TRACK1_SUBJECTS + TRACK2_SUBJECTS

    SPLITS = ('train', 'val', 'test_track1', 'test_track2')
    #: The whole release: 19 + 5 contest subjects and both test sets.
    DEFAULT_SPLITS = {'train': 19, 'val': 5, 'test_track1': 5, 'test_track2': 3}

    #: component -> (.mat variable, NIfTI-MRS file suffix). Training subjects
    #: ship "meta", "nuisance" and "all" as NIfTI too, and reading those is far
    #: cheaper than pulling a variable out of a 2.2 GB HDF5 file. "nifti_paths"
    #: checks per subject which files are actually there, so a suffix here is
    #: a name to look for, not a promise: no subject ships "mm" as NIfTI yet.
    COMPONENTS = {
        'meta':     ('xtMeta',     'mrs_fids_metabolites'),
        'mm':       ('xtMM',       'mrs_fids_macromolecules'),
        'baseline': ('xtBaseline', None),
        'nuisance': ('xtNuisance', 'mrs_fids_nuisance'),
        'all':      ('xtAll',      'mrs_fids_si_data'),
    }

    #: Named signals, each an expression over "COMPONENTS".
    PRESETS = {
        'clean':          'meta',
        'metabolites':    'meta',
        'macromolecules': 'mm',
        'composite':      'all',
        'nuisance':       'nuisance',
        'nuisance_free':  'all-nuisance',          # metab + MM + baseline + noise
    }

    def __init__(self,
                 data_dir: str = 'data/mrsi_challenge',
                 signal: str = 'clean',
                 source: str = 'mat',
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True,
                 download: bool = True,
                 dtype: Any = np.complex64):
        """
        data_dir: where the release lives, or is fetched to.
        signal: what to load, an expression over "COMPONENTS" or a "PRESETS"
                name; the default for every "load" that does not say otherwise.
        source: where to read the spectral data from — "'mat'" (default),
                "'nifti'", or "'auto'" (NIfTI where shipped, else .mat).

                The two sources hold identical data (checked on the release:
                the .mat and NIfTI arrays agree exactly once brought onto the
                same grid, which "_to_nifti_order" does). The ".mat" is the
                default because it also records the acquisition ("hzpppm",
                "ppmoff", "t") and the aux maps, so one file answers for
                everything, and it reads the test subjects' ground truth too.
                The NIfTI files are cheaper to read but omit the ppm offset.

                Test subjects ship only the composite as NIfTI, so anything else
                there comes from the ground-truth ".mat" regardless.
        cache_dir: where extracted volumes are cached. Defaults to
                "<data_dir>/_augmentrum_cache".
        use_cache: set False to always re-read the source files.
        download: fetch subjects from Zenodo when "load" finds them missing.
                Off, a missing subject is an error naming what to fetch.
        dtype: complex dtype for the cached volumes. complex64 halves both the
                cache size and the per-batch memory at no meaningful precision
                cost for this data.
        """
        self.parse_signal(signal)                   # raises on anything unknown
        if source not in ('auto', 'nifti', 'mat'):
            raise ValueError(f"source must be 'auto', 'nifti' or 'mat', got {source!r}")
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.signal = signal
        self.source = source
        self.use_cache = use_cache
        self.download = download
        self.dtype = dtype
        self.cache_dir = cache_dir or os.path.join(self.data_dir, '_augmentrum_cache')
        self._acquisition = None

    #**************#
    #   subjects   #
    #**************#
    @classmethod
    def resolve(cls, splits: Optional[Dict[str, Union[int, Sequence[str]]]] = None
                ) -> Dict[str, Tuple[str, ...]]:
        """
        Expand a splits spec into subject names per split.

        Each value is a count or an explicit sequence of subjects. Counts take
        subjects in release order; "train" and "val" both draw from the contest
        subjects, train first, so "{'train': 6, 'val': 2}" is Sub1-6 and Sub7-8.
        Test splits draw from their own pinned subjects. A split not mentioned,
        or given a count of 0, is left out entirely. None is "DEFAULT_SPLITS",
        the whole release.
        """
        splits = cls.DEFAULT_SPLITS if splits is None else splits
        unknown = set(splits) - set(cls.SPLITS)
        if unknown:
            raise ValueError(f"Unknown split(s) {sorted(unknown)}; choose from {cls.SPLITS}.")

        contest = list(cls.TRAIN_SUBJECTS)
        pools = {'test_track1': cls.TRACK1_SUBJECTS, 'test_track2': cls.TRACK2_SUBJECTS}
        resolved = {}
        for split in cls.SPLITS:
            if split not in splits:
                continue
            spec = splits[split]
            pool = pools.get(split, contest)
            if isinstance(spec, int):
                if not 0 <= spec <= len(pool):
                    raise ValueError(
                        f"splits[{split!r}]={spec}, but only {len(pool)} subjects are "
                        f"available for it."
                    )
                chosen = tuple(pool[:spec])
            else:
                chosen = tuple(spec)
                bad = [s for s in chosen if s not in pool]
                if bad:
                    raise ValueError(
                        f"splits[{split!r}] names {bad}, which are not available for "
                        f"it (already used by another split, or not its subjects)."
                    )
            if split in ('train', 'val'):
                contest = [s for s in contest if s not in chosen]
            if chosen:
                resolved[split] = chosen
        return resolved

    #**************#
    #   fetching   #
    #**************#
    def available(self, subject: str) -> bool:
        """Whether *subject* has been unpacked into "data_dir"."""
        return os.path.isfile(os.path.join(self.data_dir, subject, f'{subject}_all.mat'))

    @classmethod
    def fetch(cls, subjects: Sequence[str], root: str = 'data/mrsi_challenge',
              progress: bool = True) -> None:
        """
        Download and unpack *subjects* from Zenodo into *root*.

        Subjects already there are skipped. Each zip is checked against the
        size and MD5 the record publishes, unpacked, and removed.

        Raises:
            ValueError: If a name is not a subject of the release.
            RuntimeError: If a download does not match its published checksum.
        """
        import zipfile
        from augmentrum.utils.download import fetch

        unknown = [s for s in subjects if s not in cls.ALL_SUBJECTS]
        if unknown:
            raise ValueError(f"Not subjects of the MRSI Challenge: {unknown}")

        root = os.path.abspath(os.path.expanduser(root))
        missing = [s for s in subjects
                   if not os.path.isfile(os.path.join(root, s, f'{s}_all.mat'))]
        if not missing:
            return

        files = cls._record_files()
        total = sum(files[f'{s}.zip'][1] for s in missing) / 1e9
        if progress:
            print(f"Fetching {len(missing)} MRSI Challenge subject(s), {total:.1f} GB, "
                  f"from doi:{cls.DOI}\n"
                  f"  License: {cls.LICENSE}\n"
                  f"  Please cite: {cls.CITATION}", flush=True)

        os.makedirs(root, exist_ok=True)
        for subject in missing:
            url, size, md5 = files[f'{subject}.zip']
            archive = Path(root) / f'{subject}.zip'
            if progress:
                print(f"  {subject} ({size / 1e9:.1f} GB)", flush=True)
            fetch(url, archive, md5=md5, size=size, progress=progress)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(root)
            archive.unlink()

    @classmethod
    def _record_files(cls) -> Dict[str, Tuple[str, int, str]]:
        """Every file of the Zenodo record: name -> (url, size in bytes, md5)."""
        import json
        import urllib.request

        with urllib.request.urlopen(f'https://zenodo.org/api/records/{cls.RECORD}',
                                    timeout=60) as response:
            record = json.load(response)
        return {f['key']: (f['links']['self'], f['size'], f['checksum'].removeprefix('md5:'))
                for f in record['files']}

    #*********************#
    #   path resolution   #
    #*********************#
    def mat_path(self, subject: str, need_truth: bool) -> str:
        """
        Locate the .mat file holding *subject*.

        Test subjects ship two: the participant file, which has only the
        composite "xtAll", and the ground-truth file, which has the separated
        components. *need_truth* selects between them. Training subjects have
        everything in one file.
        """
        if subject not in self.ALL_SUBJECTS:
            raise ValueError(f"Unknown subject {subject!r}")
        truth = need_truth and subject not in self.TRAIN_SUBJECTS
        name = f'{subject}_all_truth.mat' if truth else f'{subject}_all.mat'
        return os.path.join(self.data_dir, subject, name)

    #*************#
    #   signals   #
    #*************#
    @classmethod
    def parse_signal(cls, signal: str) -> Tuple[Tuple[int, str], ...]:
        """
        Expand *signal* into signed components, e.g. "'all-nuisance'" into
        "((+1, 'all'), (-1, 'nuisance'))". Presets resolve first.
        """
        expr = cls.PRESETS.get(signal, signal).replace(' ', '')
        if not re.fullmatch(r'[+-]?[a-z]+([+-][a-z]+)*', expr):
            raise ValueError(
                f"signal must be a preset ({', '.join(cls.PRESETS)}) or an expression "
                f"over {', '.join(cls.COMPONENTS)} such as 'meta+mm', got {signal!r}"
            )
        terms = tuple((-1 if sign == '-' else 1, name)
                      for sign, name in re.findall(r'([+-]?)([a-z]+)', expr))
        unknown = [name for _, name in terms if name not in cls.COMPONENTS]
        if unknown:
            raise ValueError(
                f"signal {signal!r} names {unknown}; the components are "
                f"{', '.join(cls.COMPONENTS)}."
            )
        return terms

    @classmethod
    def canonical_signal(cls, signal: str) -> str:
        """The expression form of *signal*, the same for a preset and its expansion."""
        return ''.join(f"{'-' if sign < 0 else '+'}{name}"
                       for sign, name in cls.parse_signal(signal)).lstrip('+')

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

        The ".mat" records everything except voxel size: "hzpppm" is the
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
        """Acquisition parameters, read once from the first subject loaded."""
        if self._acquisition is None:
            present = [s for s in self.ALL_SUBJECTS if self.available(s)]
            self._acquisition = self.read_acquisition((present or self.TRAIN_SUBJECTS)[0])
        return self._acquisition

    @classmethod
    def _to_nifti_order(cls, arr: np.ndarray) -> np.ndarray:
        """
        Bring a ".mat" array onto the grid of the shipped NIfTI files.

        MATLAB stores the volumes as (Y, X, Z) with any spectral or metabolite
        axis last, and h5py reads that reversed: a 64x64x32x384 array arrives as
        (384, 32, 64, 64) = (T, Z, X, Y). The NIfTI files are (X, Y, Z, T) and
        mirrored along Y and Z relative to that - verified on the release, where
        the two sources are identical after this transform. Using one grid for
        both means "VOXEL_MM" and the aux maps hold for either source.
        """
        spatial = np.moveaxis(arr, (-2, -1, -3), (0, 1, 2))     # (X, Y, Z, rest...)
        return np.flip(spatial, axis=(1, 2))

    def nifti_paths(self, subject: str, signal: Optional[str] = None
                    ) -> Optional[List[Tuple[int, str]]]:
        """
        Signed paths to the NIfTI-MRS files making up *signal*, or None if this
        subject does not ship every component of it as NIfTI.
        """
        if subject not in self.ALL_SUBJECTS:
            raise ValueError(f"Unknown subject {subject!r}")
        folder = os.path.join(self.data_dir, subject)
        paths = []
        for sign, name in self.parse_signal(signal or self.signal):
            suffix = self.COMPONENTS[name][1]
            if suffix is None:
                return None
            paths.append((sign, os.path.join(folder, f'{subject}_{suffix}.nii.gz')))
        return paths if all(os.path.isfile(p) for _, p in paths) else None

    def read_component_nifti(self, subject: str, signal: Optional[str] = None) -> np.ndarray:
        """Read *signal* from the shipped NIfTI-MRS files."""
        import nibabel as nib

        signal = signal or self.signal
        paths = self.nifti_paths(subject, signal)
        if paths is None:
            raise FileNotFoundError(
                f"{subject} does not ship signal={signal!r} as NIfTI-MRS."
            )
        arr = 0
        for sign, path in paths:
            arr = arr + sign * np.asarray(nib.load(path).dataobj).astype(self.dtype)
        return np.ascontiguousarray(arr)

    def read_component(self, subject: str, signal: Optional[str] = None) -> np.ndarray:
        """
        Read *signal* (default: the module's) for *subject* as (X, Y, Z, T).

        Honours "source": NIfTI when available and permitted, otherwise the .mat.

        Raises:
            KeyError: If the subject's file lacks a component of the signal.
        """
        signal = signal or self.signal
        if self.source in ('auto', 'nifti'):
            if self.nifti_paths(subject, signal) is not None:
                return self.read_component_nifti(subject, signal)
            if self.source == 'nifti':
                raise FileNotFoundError(
                    f"source='nifti' but {subject} does not ship signal="
                    f"{signal!r} as NIfTI-MRS. Test subjects ship only the "
                    f"composite; use source='auto' to fall back to the .mat."
                )

        import h5py

        terms = self.parse_signal(signal)
        path = self.mat_path(subject, need_truth=self._needs_truth(terms))

        with h5py.File(path, 'r') as f:
            missing = [name for _, name in terms if self.COMPONENTS[name][0] not in f]
            if missing:
                raise KeyError(
                    f"{os.path.basename(path)} has no {missing} for signal={signal!r}; "
                    f"it contains {sorted(k for k in f.keys() if not k.startswith('#'))}."
                )
            arr = 0
            for sign, name in terms:
                arr = arr + sign * self._to_complex(f[self.COMPONENTS[name][0]])

        return np.ascontiguousarray(self._to_nifti_order(arr).astype(self.dtype))

    @staticmethod
    def _needs_truth(terms) -> bool:
        """Anything beyond the composite lives in the test subjects' ground-truth file."""
        return any(name != 'all' for _, name in terms)

    def read_aux(self, subject: str, signal: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Read the per-subject auxiliary maps: brain mask, B0 map, anatomical
        reference and — where available — the ground-truth metabolite amplitudes.

        All are returned in (X, Y, Z) order to match the spectral volumes.
        "metaMap" gains a trailing metabolite axis: (X, Y, Z, n_metabolites).
        """
        import h5py

        terms = self.parse_signal(signal or self.signal)
        path = self.mat_path(subject, need_truth=self._needs_truth(terms))
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
    def _cache_path(self, subject: str, signal: str) -> str:
        return os.path.join(self.cache_dir, f'{subject}_{self.canonical_signal(signal)}.nii')

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

    def load_subject(self, subject: str, signal: Optional[str] = None):
        """
        Return one subject's *signal* as a NIfTI-MRS object, cached when possible.

        The cache is written uncompressed so "read_FID" can memory-map it; a
        gzipped file would have to be inflated into RAM in full.
        """
        from fsl_mrs.utils import mrs_io

        signal = signal or self.signal
        cache = self._cache_path(subject, signal)
        if self.use_cache and os.path.isfile(cache):
            return mrs_io.read_FID(cache)

        nifti = self.to_nifti(self.read_component(subject, signal))

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            nifti.save(cache)
            return mrs_io.read_FID(cache)        # reopen memory-mapped
        return nifti

    def load(self, subjects: Sequence[str], signal: Optional[str] = None,
             with_aux: bool = False) -> Tuple[List, List[Dict]]:
        """
        Load *subjects*, fetching any that are missing when "download" is on.

        Args:
            subjects: names such as "'Sub3'" or "'TestSub10'"; see "resolve" for
                turning a splits spec into these.
            signal: what to load, defaulting to the module's; see "COMPONENTS".
            with_aux: also read the auxiliary maps (brain mask, B0, metaMap).

        Returns:
            (nifti_list, aux_list), in the order given. "aux_list" is a list of
            empty dicts when *with_aux* is False.
        """
        subjects = list(subjects)
        missing = [s for s in subjects if not self.available(s)]
        if missing and not self.download:
            raise FileNotFoundError(
                f"Not in {self.data_dir}: {missing}. Fetch them with\n"
                f"  MRSIChallengeDataModule.fetch({missing!r}, {self.data_dir!r})\n"
                f"or construct with download=True."
            )
        if missing:
            self.fetch(subjects, self.data_dir)

        data, aux = [], []
        for name in subjects:
            data.append(self.load_subject(name, signal))
            aux.append(self.read_aux(name, signal) if with_aux else {})
        return data, aux

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
        here.
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


def MRSIChallengeData(data_dir: str = 'data/mrsi_challenge',
                      signal: Union[str, Dict[str, str]] = 'clean',
                      source: str = 'mat',
                      splits: Optional[Dict[str, Union[int, Sequence[str]]]] = None,
                      batch_size: int = 2,
                      seed: int = 42,
                      baseline: bool = False,
                      pipelines: Optional[Dict[str, Any]] = None,
                      modes: Optional[Dict[str, str]] = None,
                      backend: str = 'pytorch',
                      volatile: bool = True,
                      cache_dir: Optional[str] = None,
                      use_cache: bool = True,
                      download: bool = True,
                      with_aux: bool = False,
                      **kwargs) -> Augmentrum:
    """
    Load the MRSI Challenge into an "~augmentrum.core.augmentrum.Augmentrum".

    Train and validation are carved out of the 24 contest subjects; both test sets
    are the challenge's own held-out subjects and are pinned by name rather than
    sampled, so they can never leak into training. Only the subjects the splits
    name are loaded, and fetched from Zenodo if they are not there yet.

    Args:
        data_dir: where the release lives, or is fetched to.
        signal: what to load, as a sum or difference of the release's
                components ("'meta+mm'", "'all-nuisance'") or a preset; see
                "MRSIChallengeDataModule". One expression for every split, or
                a dict per split - splits it leaves out get "'clean'". The
                usual shape is "{'train': 'clean', 'test_track1': 'meta+mm'}":
                train on the noiseless metabolites, evaluate against
                metabolites plus macromolecules.
        source: "'mat'", "'nifti'" or "'auto'" (see "MRSIChallengeDataModule").
        splits: subjects per split, each a count or an explicit sequence; see
                "MRSIChallengeDataModule.resolve". None loads the whole release:
                19 train, 5 val and both test sets. "{'train': 6, 'val': 2}"
                loads eight subjects and no test set.
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
        download: fetch missing subjects from Zenodo (see "MRSIChallengeDataModule").
        with_aux: attach the per-subject brain mask, B0 map and metabolite maps to
                the returned object as ".aux" (a dict keyed by split).
        **kwargs: module parameters forwarded to Augmentrum, e.g.
                "acceleration_factor=(2.0, 6.0)" or "sigma=1e-3".

    Returns:
        Augmentrum with one split per entry of *splits*, the subject names
        behind each on ".subject_names" and the signal each holds on ".signals".

    Examples:
        >>> aug = MRSIChallengeData(splits={'train': 6, 'val': 2, 'test_track1': 5},
        ...                         signal={'test_track1': 'meta+mm'},
        ...                         acceleration_factor=(2.0, 6.0), sigma=(0.6e-3, 1.4e-3))
        >>> aug.signals
        {'train': 'clean', 'val': 'clean', 'test_track1': 'meta+mm'}
        >>> batch, _ = next(aug.train_dataloader())
        >>> batch.shape
        torch.Size([2, 64, 64, 32, 384])
    """
    module = MRSIChallengeDataModule(data_dir, source=source, cache_dir=cache_dir,
                                     use_cache=use_cache, download=download)
    subjects = module.resolve(splits)
    signals = {split: (signal.get(split, 'clean') if isinstance(signal, dict) else signal)
               for split in subjects}

    all_data, split_indices, names, aux = [], {}, {}, {}
    for split, members in subjects.items():
        data, split_aux = module.load(members, signals[split], with_aux=with_aux)
        split_indices[split] = list(range(len(all_data), len(all_data) + len(data)))
        all_data.extend(data)
        names[split] = list(members)
        aux[split] = split_aux
    if not all_data:
        raise ValueError(f"splits={splits!r} selects no subjects.")

    if pipelines is None:
        # signal='clean' is metabolites alone — no macromolecules, no baseline, no
        # noise (the release separates MM only in the test ground truth; see
        # MRSIChallengeDataModule for 'meta+mm' there). Augmentrum supplies the
        # missing realism, which is the point: it is parameterized and
        # reproducible, whereas whatever the release happens to contain is fixed.
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

    aug.subject_names = names
    aug.signals = signals
    if with_aux:
        aug.aux = aux
    aug.data_module = module
    return aug
