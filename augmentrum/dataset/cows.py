####################################################################################################
#                                             cows.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-14                                                                              #
#                                                                                                  #
# Purpose: Loads the COWS study (OpenNeuro ds006812; 3T sLASER under VAPOR / COWS water            #
#          suppression) into standard NIfTI-MRS - raw Siemens TWIX with every transient and coil,  #
#          or the INSPECTOR-processed .mat derivatives - with subject, region and suppression      #
#          scheme carried in the header extension.                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

# third-party
import numpy as np

# own
from augmentrum import Augmentrum


__all__ = ['COWSData', 'COWSDataModule', 'COWSScan', 'read_twix', 'read_mat', 'is_mat_water',
           'scan_info', 'REGIONS', 'WATER_SUPPRESSIONS', 'HEADER_FIELDS', 'PROTOCOL']


#*****************#
#   vocabulary    #
#*****************#
#: Region spellings found in the study -> canonical code. sub-10 labels its prefrontal voxel
#: 'PFC' where every other subject writes 'PFL', and the derivatives capitalise the other two.
REGIONS = {'PFL': 'PFL', 'PFC': 'PFL',
           'OCCIPITAL': 'OCC', 'OCC': 'OCC',
           'PARIETAL': 'PAR', 'PAR': 'PAR'}

#: Water suppression spellings -> canonical scheme ('VAPOR' alone means the 7-pulse VAPOR).
WATER_SUPPRESSIONS = {'VAPOR': 'vapor7', 'VAPOR7': 'vapor7', 'COWS7': 'cows7', 'COWS12': 'cows12'}

SCAN_TYPES = ('metab', 'mm')

#: The acquisition order every subject followed (verified identical across all ten): five scans
#: per region, prefrontal first, metabolite before its metabolite-nulled (MM) partner. The .mat
#: derivatives carry no acq index in their names, so it is read off this table.
PROTOCOL = {
    (region, ws, kind): 5 * r + s + 1
    for r, region in enumerate(('PFL', 'OCC', 'PAR'))
    for s, (ws, kind) in enumerate((('vapor7', 'metab'), ('vapor7', 'mm'), ('cows7', 'metab'),
                                    ('cows7', 'mm'), ('cows12', 'metab')))
}

#: Protocol constants the .mat derivatives dropped; identical in every TWIX header of the study.
ECHO_TIME = 0.026                                   # s
REPETITION_TIME = 2.0                               # s

#: User-defined NIfTI-MRS header fields every loaded scan carries, with their descriptions.
HEADER_FIELDS = {
    'SubjectID': 'BIDS subject label, e.g. sub-01',
    'Region': "Voxel region: 'PFL' (prefrontal), 'OCC' (occipital) or 'PAR' (parietal)",
    'WaterSuppression': "Water suppression scheme: 'vapor7', 'cows7' or 'cows12'",
    'ScanType': "'metab' (metabolite) or 'mm' (metabolite-nulled macromolecule) acquisition",
    'Acquisition': 'BIDS acq index of the scan within its session (1-15)',
}

_TWIX_NAME = re.compile(
    r'^(?P<subject>sub-\d+)_acq-(?P<acq>\d+)_.*?_(?P<ws>vapor7|cows7|cows12)'
    r'_(?P<kind>metab|mm)_(?P<region>[A-Za-z]+)\.dat$', re.IGNORECASE)
_MAT_NAME = re.compile(
    r'^(?P<subject>sub-\d+)_(?P<ws>vapor7|cows7|cows12)_(?P<kind>metab|mm)(?P<water>_water)?\.mat$',
    re.IGNORECASE)
_MAT_SUBJECT = re.compile(r'^sub-\d+(_mat)?$')


#**************************************************************************************************#
#                                          Class COWSScan                                          #
#**************************************************************************************************#
#                                                                                                  #
# What one file of the study is: who, where, how the water was suppressed, and which acquisition.  #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True, order=True)
class COWSScan:
    """
    Metadata of one scan, parsed from its file name.

    Ordering is by subject, then acquisition index, which is the deterministic
    file order the loader delivers.

    Args:
        subject: BIDS label, e.g. 'sub-01'.
        acquisition: BIDS acq index, 1-15.
        water_suppression: 'vapor7', 'cows7' or 'cows12'.
        scan_type: 'metab' or 'mm'.
        region: canonical region code, see "REGIONS".
        path: the file the scan is read from.
    """
    subject: str
    acquisition: int
    water_suppression: str
    scan_type: str
    region: str
    path: str

    @property
    def stem(self) -> str:
        """Unique name, e.g. 'sub-01_acq-01_vapor7_metab_PFL'; used for names and cache files."""
        return (f"{self.subject}_acq-{self.acquisition:02d}_{self.water_suppression}"
                f"_{self.scan_type}_{self.region}")

    @classmethod
    def from_twix(cls, path):
        """Parse a sourcedata TWIX name; None for a file that is not a COWS scan."""
        match = _TWIX_NAME.match(os.path.basename(path))
        if match is None or match['region'].upper() not in REGIONS:
            return None
        return cls(match['subject'].lower(), int(match['acq']), match['ws'].lower(),
                   match['kind'].lower(), REGIONS[match['region'].upper()], str(path))

    @classmethod
    def from_mat(cls, path):
        """
        Parse a derivatives .mat name ("<region>/sub-XX_<WS>_<Metab|MM>[_Water].mat").

        The region comes from the parent directory and the acquisition index
        from "PROTOCOL". None for a file that is not a COWS scan; water files
        parse like their metabolite partner, ask "is_mat_water" to tell them apart.
        """
        match = _MAT_NAME.match(os.path.basename(path))
        region = os.path.basename(os.path.dirname(path)).upper()
        if match is None or region not in REGIONS:
            return None
        ws, kind, region = match['ws'].lower(), match['kind'].lower(), REGIONS[region]
        return cls(match['subject'].lower(), PROTOCOL[(region, ws, kind)], ws, kind, region,
                   str(path))


def is_mat_water(path) -> bool:
    """Whether a derivatives .mat file is the water reference of its scan."""
    match = _MAT_NAME.match(os.path.basename(path))
    return match is not None and match['water'] is not None


#********************#
#   header fields    #
#********************#
def scan_info(nifti) -> dict:
    """
    The COWS header fields of a loaded scan as plain values.

    nifti_mrs stores user-defined fields as "{'Value': ..., 'Description': ...}"
    and "hdr_ext[key]" returns that dict; this unwraps it.

    Args:
        nifti: NIFTI_MRS object.

    Returns:
        Dict with whichever of "HEADER_FIELDS" the object carries.
    """
    info = {}
    for key in HEADER_FIELDS:
        if key in nifti.hdr_ext:
            value = nifti.hdr_ext[key]
            info[key] = value['Value'] if isinstance(value, dict) and 'Value' in value else value
    return info


def _stamp(nifti, scan: COWSScan):
    """Write the scan's metadata into the header extension; returns the same object."""
    values = {'SubjectID': scan.subject, 'Region': scan.region,
              'WaterSuppression': scan.water_suppression, 'ScanType': scan.scan_type,
              'Acquisition': int(scan.acquisition)}
    for key, doc in HEADER_FIELDS.items():
        nifti.add_hdr_field(key, values[key], doc=doc)
    return nifti


#*************#
#   readers   #
#*************#
def read_twix(path, remove_oversampling: bool = True):
    """
    Read one COWS TWIX file into standard NIfTI-MRS.

    Everything goes through spec2nii's "process_twix", so the result is what
    "spec2nii twix" writes: right-handed storage (plain "np.fft.fft" of the
    stored FID with the FSL-MRS ppm axis puts NAA at 2.01 ppm, water at 4.65)
    and the full header extension - EchoTime, RepetitionTime, TxOffset, the
    scanner and sequence fields. No conjugation, re-creation or decimation
    happens here, which is where earlier versions went wrong.

    The sequence's 'Phs' loop interleaves the unsuppressed water (index 0)
    with the water-suppressed acquisition (index 1); spec2nii tags it
    DIM_USER_0 and this splits it back out. The water sits in a DIM_DYN axis
    zero-padded to the metabolite's transient count; only the acquired
    (non-zero) transients are kept and a singleton axis is squeezed, so the
    water arrives as (1, 1, 1, T, coils) or (1, 1, 1, T, coils, n).

    Args:
        path: The .dat file.
        remove_oversampling: Remove the 2x readout oversampling (default
            True; 4096 pts / 8000 Hz -> 2048 pts / 4000 Hz). This is pymapvbvd's
            "flagRemoveOS": FFT along the readout, keep the central half, IFFT
            - a frequency-domain crop, not decimation, which would fold the
            outer half of the noise back in (1.2-1.3x higher noise SD).

    Returns:
        "(data, water)" NIFTI_MRS objects with dim tags DIM_COIL, DIM_DYN and
        DIM_COIL[, DIM_DYN].
    """
    from fsl_mrs.core.nifti_mrs import split
    from spec2nii.Siemens.twixfunctions import process_twix
    from augmentrum.processing.utils import safe_squeeze

    try:
        from mapvbvd import mapVBVD
    except ImportError as error:
        raise ImportError(
            "Reading Siemens twix files needs mapvbvd. Install it with "
            "\"pip install pymapvbvd\", or point this at data already "
            "converted to NIfTI-MRS.") from error

    twix = mapVBVD(str(path), quiet=True)
    if isinstance(twix, list):                      # multi-RAID file: the last is the scan
        twix = twix[-1]

    name = os.path.basename(path)
    overrides = {'dims': (None, None, None), 'tags': (None, None, None)}
    images, _ = process_twix(twix, os.path.splitext(name)[0], name, 'image', overrides,
                             quiet=True, remove_os=remove_oversampling)

    water, data = split(images[0], 'DIM_USER_0', 0)
    data = safe_squeeze(data, dims=['DIM_USER_0'])
    water = _acquired_water(safe_squeeze(water, dims=['DIM_USER_0']))
    return data, water


def _acquired_water(water):
    """Keep the leading non-zero DIM_DYN entries of the zero-padded water; squeeze if one."""
    from fsl_mrs.core.nifti_mrs import split
    from augmentrum.processing.utils import safe_squeeze

    if 'DIM_DYN' not in water.dim_tags:
        return water
    axis = water.dim_position('DIM_DYN')
    array = np.asarray(water[:]).reshape(water.shape)
    others = tuple(a for a in range(array.ndim) if a != axis)
    acquired = np.flatnonzero(np.abs(array).max(axis=others) > 0)
    if acquired.size == 0:
        raise ValueError('the water reference holds no non-zero transient')

    last = int(acquired.max())
    if last + 1 < water.shape[axis]:
        water, _ = split(water, 'DIM_DYN', last)
    return safe_squeeze(water, dims=['DIM_DYN'])


def read_mat(path):
    """
    Read one INSPECTOR .mat derivative into NIfTI-MRS.

    These are already processed: one coil-combined, averaged FID per file
    ("exptDat.fid" with "sf", "sw_h", "nspecC"), so the result has no
    DIM_COIL / DIM_DYN and needs no coil combination or averaging.

    INSPECTOR stores the FID in the FSL-MRS user-facing (left-handed)
    convention - what "NIFTI_MRS[:]" returns - so "gen_nifti_mrs" must apply
    its default conjugation on creation. Verified against the TWIX-derived
    spectrum of the same scan: with it NAA / Cr / water sit at 1.98 / 3.01 /
    4.64 ppm (INSPECTOR's referencing is ~4 Hz off spec2nii's), with
    "no_conj=True" the spectrum is mirrored. EchoTime and RepetitionTime, which
    the .mat omits, are filled from the study protocol.

    Args:
        path: The .mat file.

    Returns:
        NIFTI_MRS object of shape (1, 1, 1, T).
    """
    import scipy.io
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    record = scipy.io.loadmat(str(path))['exptDat'][0, 0]
    fid = np.asarray(record['fid']).ravel().astype(np.complex128)
    nifti = gen_nifti_mrs(fid.reshape(1, 1, 1, -1),
                          dwelltime=1.0 / float(np.squeeze(record['sw_h'])),
                          spec_freq=float(np.squeeze(record['sf'])),
                          nucleus='1H',
                          dim_tags=[None, None, None])
    nifti.add_hdr_field('EchoTime', ECHO_TIME)
    nifti.add_hdr_field('RepetitionTime', REPETITION_TIME)
    nifti.add_hdr_field('OriginalFile', [os.path.basename(path)])
    return nifti


#*****************************#
#   worker / cache transport   #
#*****************************#
def _to_bytes(nifti) -> bytes:
    """Serialise a NIFTI_MRS to NIfTI-2 bytes (header extension and affine included)."""
    nifti.hdr_ext = nifti.hdr_ext                   # re-embeds the extension in the image
    return nifti.image.nibImage.to_bytes()


def _from_bytes(blob: bytes):
    import nibabel as nib
    from fsl_mrs.core.nifti_mrs import NIFTI_MRS

    return NIFTI_MRS(nib.Nifti2Image.from_bytes(blob))


def _load_twix_scan(scan: COWSScan, remove_oversampling: bool):
    """One TWIX scan, stamped with its metadata."""
    data, water = read_twix(scan.path, remove_oversampling)
    return _stamp(data, scan), _stamp(water, scan)


def _load_twix_scan_bytes(args):
    """
    Worker entry point.

    NIFTI_MRS objects do not pickle (they hold a weakref), so the worker hands
    back serialised NIfTI-2 bytes and the parent rebuilds the objects.
    """
    scan, remove_oversampling = args
    data, water = _load_twix_scan(scan, remove_oversampling)
    return _to_bytes(data), _to_bytes(water)


#**********************#
#   cows data loader   #
#**********************#
def COWSData(data_dir, batch_size=16, seed=0, val_frac=0.1, test_frac=0.1,
             n_coils=(1, None), n_averages=(1, None), pipelines=None,
             modes=None, backend='pytorch', volatile=False,
             location=None, water_sup=None, subjects=None, source='twix',
             remove_oversampling=True, cache_dir=None, workers=None, strict=True,
             device=None, compress_cache=False, **kwargs):
    """
    Load COWS metabolite scans and create an Augmentrum instance.

    The default training pipeline draws coils and transients per sample (as
    masks the processing consumes) and, on tensor backends, processes with the
    batched torch engine (registration_method='torch'), the subject pool
    stacked once on *device*; pass registration_method to choose another.

    Args:
        data_dir: Path to the COWS data directory (the ds006812 root).
        batch_size: Batch size for dataloaders.
        seed: Random seed for reproducibility.
        val_frac: Validation fraction (default 0.1).
        test_frac: Test fraction (default 0.1).
        n_coils: Coil sampling range (min, max) or None.
        n_averages: Average sampling range (min, max) or None.
        pipelines: Custom pipelines dict or None for defaults.
        modes: Sampling modes dict or None for defaults.
        backend: Backend to use ('numpy', 'pytorch', etc.).
        volatile: If True, skip provenance logging.
        location: Region(s) to keep, see "COWSDataModule". None keeps all.
        water_sup: Water suppression scheme(s) to keep. None keeps all.
        subjects: Subject labels to keep, e.g. ('sub-01',). None keeps all.
        source: 'twix' (raw, default) or 'mat' (INSPECTOR-processed derivatives).
        remove_oversampling: Remove the 2x readout oversampling (TWIX only).
        cache_dir: Where loaded scans are cached as NIfTI-MRS (TWIX only).
        compress_cache: Write that cache gzipped, see "COWSDataModule".
        workers: Processes to read TWIX files with (None or 1: in-process).
        strict: Raise on a scan that fails to load; False skips it with a warning.
        device: Torch device for the pooled subjects and batches (None: CPU).
        **kwargs: Additional parameters for modules.

    Returns:
        Augmentrum instance with COWS data loaded.
    """
    if pipelines is None:
        # Train on random coil and transient subsets of the raw acquisition - a
        # subset of its own for every sample - then process; validate and test
        # on the full acquisition, processed the same way.
        pipelines = {'train': [{'coil_sampling': {'per_sample': True}},
                               {'average_sampling': {'per_sample': True}}, 'processing'],
                     'val': ['processing'], 'test': ['processing']}
    if modes is None:
        modes = {'train': 'on-the-fly', 'val': 'fixed', 'test': 'fixed'}

    # The sampling ranges only reach a pipeline that draws; with the samplers
    # absent, Augmentrum would rightly refuse them as unknown kwargs.
    accepted = Augmentrum.accepted_parameters(pipelines)
    sampling = {key: value for key, value in (('n_coils', n_coils), ('n_averages', n_averages))
                if key in accepted}
    # Tensor batches are processed by the batched engine unless told otherwise.
    if (str(getattr(backend, 'value', backend)).lower() != 'nifti_list'
            and 'registration_method' in accepted
            and 'registration_method' not in kwargs):
        kwargs['registration_method'] = 'torch'

    loader = COWSDataModule(data_dir=data_dir, location=location, water_sup=water_sup,
                            subjects=subjects, remove_oversampling=remove_oversampling,
                            cache_dir=cache_dir, workers=workers, strict=strict,
                            compress_cache=compress_cache)
    if source == 'twix':
        data, water, _, _, _ = loader.load_twix()
    elif source == 'mat':
        data, water, _, _, _ = loader.load_mats()
    else:
        raise ValueError(f"source must be 'twix' or 'mat', got {source!r}")

    # Split by subject: the scans of one person never straddle train and validation.
    groups = [scan_info(nifti)['SubjectID'] for nifti in data]

    return Augmentrum(
        data=data,
        water=water,
        split_fractions={'val': val_frac, 'test': test_frac},
        pipelines=pipelines,
        modes=modes,
        backend=backend,
        batch_size=batch_size,
        seed=seed,
        volatile=volatile,
        groups=groups,
        device=device,
        **sampling,
        **kwargs
    )


#**************************************************************************************************#
#                                       Class COWSDataModule                                       #
#**************************************************************************************************#
#                                                                                                  #
# Walks the ds006812 tree and delivers each scan as NIfTI-MRS, from raw TWIX or the .mat           #
# derivatives, in a deterministic order with its metadata in the header.                          #
#                                                                                                  #
#**************************************************************************************************#
class COWSDataModule:
    """
    Loader for the COWS study (OpenNeuro ds006812).

    Ten subjects, three voxels each (prefrontal, occipital, parietal), and per
    voxel five sLASER scans (TE 26 ms, TR 2 s, 3T Prisma, 32-channel coil):
    metabolite and metabolite-nulled (MM) acquisitions under VAPOR, and under
    the 7- and 12-pulse COWS schemes (COWS12 has no MM partner). The
    unsuppressed water reference is inside each TWIX file.

    Two sources:

    - "load_twix": raw "sub-XX/mrs/sourcedata/*.dat", read through spec2nii
      into standard NIfTI-MRS with every coil and transient,
      (1, 1, 1, 2048, 32, 32 | 64) at 4000 Hz, and the water reference with
      all of its acquired transients, (1, 1, 1, 2048, 32[, n]).
    - "load_mats": "derivatives/mrs_mat/sub-XX_mat/<REGION>/*.mat", one
      already processed (coil-combined, averaged) FID per scan, (1, 1, 1, 2048).

    Both stamp "HEADER_FIELDS" (subject, region, suppression, scan type,
    acquisition) into the header extension; "scan_info" reads them back.
    Files are delivered sorted by subject then acquisition, and the names
    returned line up with the data (metabolite) lists; the MM lists have
    their own "mm_names". "scans" / "mm_scans" hold the "COWSScan" records.

    Region names are canonical - 'PFL', 'OCC', 'PAR' - and every spelling in
    the study ('PFC', 'Occipital', 'PARIETAL', ...) maps onto them, see
    "REGIONS".

    Caching: with "cache_dir" each TWIX scan is written once as
    "<stem>.nii" and "<stem>_water.nii" plus an "index.json", and read from
    there afterwards, header fields included. Uncompressed by default: raw
    coil and transient data is mostly noise, so gzip saves ~7 % of the space
    while every read has to inflate the whole file (~60 ms a scan against
    ~3 ms memory-mapped). Reading TWIX is ~0.35 s a file and needs pymapvbvd.

    Args:
        data_dir: The ds006812 root (or, for "load_mats", the "mrs_mat"
            directory itself).
        location: Region(s) to keep, any spelling in "REGIONS", string or
            iterable. None keeps all.
        water_sup: Suppression scheme(s) to keep, any spelling in
            "WATER_SUPPRESSIONS". None keeps all.
        subjects: Subject labels to keep, e.g. ('sub-01', 'sub-02'). None
            keeps all.
        remove_oversampling: Remove the 2x readout oversampling when reading
            TWIX (default True), see "read_twix".
        cache_dir: Directory for the NIfTI-MRS cache. None disables caching.
        workers: Number of processes to read TWIX files with. None or 1
            reads in-process.
        strict: Raise on the first scan that fails to load (default). False
            skips it with a warning and records it in "load_failures".
        compress_cache: Write the cache as ".nii.gz" instead of ".nii". A
            cache in the other format is still read rather than rebuilt.
    """

    INDEX = 'index.json'

    def __init__(self, data_dir, location=None, water_sup=None, subjects=None,
                 remove_oversampling: bool = True, cache_dir=None, workers=None,
                 strict: bool = True, compress_cache: bool = False):
        self.data_dir = str(data_dir)
        self.regions = self._canonical(location, REGIONS, 'location')
        self.water_sup = self._canonical(water_sup, WATER_SUPPRESSIONS, 'water_sup')
        self.subjects = None if subjects is None else tuple(
            s.lower() for s in ([subjects] if isinstance(subjects, str) else subjects))
        self.remove_oversampling = bool(remove_oversampling)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.compress_cache = bool(compress_cache)
        self.workers = workers
        self.strict = strict

        self.load_failures = []                     # (path, error message)
        self.scans, self.mm_scans = [], []
        self.names, self.mm_names = [], []

    @staticmethod
    def _canonical(values, table, what):
        """Normalise a filter to canonical codes; None means every code in the table."""
        if values is None:
            return tuple(sorted(set(table.values())))
        if isinstance(values, str):
            values = [values]
        unknown = [v for v in values if str(v).upper() not in table]
        if unknown:
            raise ValueError(f"unknown {what} {unknown}; choose from {sorted(table)}")
        return tuple(sorted({table[str(v).upper()] for v in values}))

    def _wanted(self, scan: COWSScan) -> bool:
        return (scan.region in self.regions and scan.water_suppression in self.water_sup
                and (self.subjects is None or scan.subject in self.subjects))

    def _fail(self, scan: COWSScan, error: Exception, strict: bool):
        message = f"Failed to load {scan.path}: {error}"
        if strict:
            raise RuntimeError(message) from error
        warnings.warn(message)
        self.load_failures.append((scan.path, str(error)))

    def _deliver(self, loaded: dict):
        """Split loaded scans into the metabolite and MM lists, in scan order."""
        data, water, names, self.scans = [], [], [], []
        mm, mm_water, mm_names, self.mm_scans = [], [], [], []
        for scan in sorted(loaded):
            met, ref = loaded[scan]
            if scan.scan_type == 'metab':
                data.append(met); water.append(ref); names.append(scan.stem)
                self.scans.append(scan)
            else:
                mm.append(met); mm_water.append(ref); mm_names.append(scan.stem)
                self.mm_scans.append(scan)
        self.names, self.mm_names = names, mm_names
        return data, water, mm, mm_water, names

    #**********#
    #   twix   #
    #**********#
    def twix_scans(self):
        """The TWIX scans matching the filters, sorted by subject then acquisition."""
        scans = []
        for entry in sorted(os.listdir(self.data_dir)):
            source = os.path.join(self.data_dir, entry, 'mrs', 'sourcedata')
            if not (entry.startswith('sub-') and os.path.isdir(source)):
                continue
            for name in os.listdir(source):
                scan = COWSScan.from_twix(os.path.join(source, name))
                if scan is not None and self._wanted(scan):
                    scans.append(scan)
        return sorted(scans)

    def load_twix(self, strict=None):
        """
        Load every matching TWIX scan.

        Args:
            strict: Overrides the instance setting for this call.

        Returns:
            "(data, water, mm, mm_water, names)" - lists of NIFTI_MRS objects;
            "names" lines up with "data", "self.mm_names" with "mm".
        """
        strict = self.strict if strict is None else strict
        self.load_failures = []
        scans = self.twix_scans()

        loaded, pending = {}, []
        index = self._index() if self.cache_dir else {}
        for scan in scans:
            cached = self._read_cache(scan, index) if self.cache_dir else None
            if cached is None:
                pending.append(scan)
            else:
                loaded[scan] = cached

        if self.workers and self.workers > 1 and pending:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures = [(scan, pool.submit(_load_twix_scan_bytes,
                                              (scan, self.remove_oversampling)))
                           for scan in pending]
                for scan, future in futures:
                    try:
                        loaded[scan] = tuple(_from_bytes(blob) for blob in future.result())
                    except Exception as error:
                        self._fail(scan, error, strict)
        else:
            for scan in pending:
                try:
                    loaded[scan] = _load_twix_scan(scan, self.remove_oversampling)
                except Exception as error:
                    self._fail(scan, error, strict)

        if self.cache_dir and any(scan in loaded for scan in pending):
            for scan in pending:
                if scan in loaded:
                    self._write_cache(scan, *loaded[scan], index)
            self._save_index(index)

        return self._deliver(loaded)

    #***********#
    #   cache   #
    #***********#
    def _cache_paths(self, scan: COWSScan, compressed=None):
        # Oversampled data is a different array, so it gets its own files.
        stem = scan.stem + ('' if self.remove_oversampling else '_os')
        ext = '.nii.gz' if (self.compress_cache if compressed is None else compressed) else '.nii'
        return (os.path.join(self.cache_dir, stem + ext),
                os.path.join(self.cache_dir, stem + '_water' + ext))

    def _index(self) -> dict:
        path = os.path.join(self.cache_dir, self.INDEX)
        if not os.path.isfile(path):
            return {}
        with open(path) as handle:
            return json.load(handle)

    def _save_index(self, index: dict):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(os.path.join(self.cache_dir, self.INDEX), 'w') as handle:
            json.dump(index, handle, indent=1, sort_keys=True)

    def _read_cache(self, scan: COWSScan, index: dict):
        """The cached "(data, water)" of a scan, or None if absent or from another source."""
        from fsl_mrs.utils.mrs_io import read_FID

        # the configured format first, then a cache written in the other one
        for compressed in (self.compress_cache, not self.compress_cache):
            data_path, water_path = self._cache_paths(scan, compressed)
            entry = index.get(os.path.basename(data_path))
            if (entry is not None and os.path.isfile(data_path) and os.path.isfile(water_path)
                    and entry.get('size') == os.path.getsize(scan.path)):
                return read_FID(data_path), read_FID(water_path)
        return None

    def _write_cache(self, scan: COWSScan, data, water, index: dict):
        """Save a scan's NIfTIs and record them in "index" (the caller saves the index)."""
        os.makedirs(self.cache_dir, exist_ok=True)
        data_path, water_path = self._cache_paths(scan)
        data.save(data_path)
        water.save(water_path)

        index[os.path.basename(data_path)] = {
            'water': os.path.basename(water_path),
            'source': os.path.basename(scan.path),
            'size': os.path.getsize(scan.path),
            'remove_oversampling': self.remove_oversampling,
            'subject': scan.subject, 'acquisition': scan.acquisition,
            'region': scan.region, 'water_suppression': scan.water_suppression,
            'scan_type': scan.scan_type,
        }

    #*********#
    #   mat   #
    #*********#
    def mat_scans(self):
        """The .mat scans (metabolite / MM files only) matching the filters, sorted."""
        root = os.path.join(self.data_dir, 'derivatives', 'mrs_mat')
        if not os.path.isdir(root):
            root = self.data_dir

        scans = []
        for entry in sorted(os.listdir(root)):
            subject_dir = os.path.join(root, entry)
            if not (_MAT_SUBJECT.match(entry) and os.path.isdir(subject_dir)):
                continue
            for region in sorted(os.listdir(subject_dir)):
                region_dir = os.path.join(subject_dir, region)
                if region.upper() not in REGIONS or not os.path.isdir(region_dir):
                    continue
                for name in os.listdir(region_dir):
                    path = os.path.join(region_dir, name)
                    scan = COWSScan.from_mat(path)
                    if scan is not None and not is_mat_water(path) and self._wanted(scan):
                        scans.append(scan)
        return sorted(scans)

    def load_mats(self, strict=None):
        """
        Load every matching INSPECTOR .mat scan, see "read_mat".

        The water reference is the "_Water.mat" next to each scan; a scan
        without one gets a None water entry.

        Args:
            strict: Overrides the instance setting for this call.

        Returns:
            "(data, water, mm, mm_water, names)" as "load_twix".
        """
        strict = self.strict if strict is None else strict
        self.load_failures = []

        loaded = {}
        for scan in self.mat_scans():
            water_path = scan.path[:-4] + '_Water.mat'
            try:
                met = _stamp(read_mat(scan.path), scan)
                ref = _stamp(read_mat(water_path), scan) if os.path.isfile(water_path) else None
            except Exception as error:
                self._fail(scan, error, strict)
                continue
            loaded[scan] = (met, ref)

        return self._deliver(loaded)


#*************#
#   testing   #
#*************#
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # NIfTI objects in, NIfTI objects out: the list backend runs the FSL-MRS processing path
    cows = COWSData(data_dir='data/openneuro_ds006812/', location='PFL', water_sup='vapor7',
                    cache_dir='data/openneuro_ds006812/_augmentrum_cache',
                    backend='nifti_list', conj=False, coil_method='fsl-mrs')

    # example
    x, x_ref = next(iter(cows.train_dataloader()))
    for elem in x:
        print(scan_info(elem))
        elem.plot()
        plt.show()
