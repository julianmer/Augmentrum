####################################################################################################
#                                          test_cows.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-16                                                                              #
#                                                                                                  #
# Purpose: Holds the COWS loader to delivering standard NIfTI-MRS in a deterministic order, with   #
#          the scan's identity in the header, from raw TWIX (cached or not) and the .mat           #
#          derivatives alike.                                                                      #
#                                                                                                  #
####################################################################################################

"""
Tests for the COWS loader.

The TWIX files are 70 MB each, so everything around reading them is exercised
on stand-ins: name parsing and the canonical vocabulary, the order and the
aligned names, the header fields and their read-back, the cache, the worker
transport, failure handling and the .mat orientation. The real files are
touched only by the slow tests, which run when the study is on disk.
"""

#*************#
#   imports   #
#*************#
import json
import multiprocessing
import os
import warnings
import zlib

import numpy as np
import pytest
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

from augmentrum.dataset import cows
from augmentrum.dataset.cows import (COWSDataModule, COWSScan, HEADER_FIELDS, PROTOCOL,
                                     is_mat_water, read_mat, scan_info)


DATA_DIR = os.environ.get(
    'COWS_DATA_DIR',
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'openneuro_ds006812'))

#: A stand-in study: sub-10 spelling its prefrontal voxel 'PFC', files out of order on disk.
FILES = [
    ('sub-10', 'sub-10_acq-01_svs_slaser_vapor7_metab_PFC.dat'),
    ('sub-10', 'sub-10_acq-02_svs_slaser_vapor7_mm_PFC.dat'),
    ('sub-01', 'sub-01_acq-13_svs_slaser_cows7_metab_Parietal.dat'),
    ('sub-01', 'sub-01_acq-06_svs_slaser_vapor7_metab_Occipital.dat'),
    ('sub-01', 'sub-01_acq-14_svs_slaser_cows7_mm_Parietal.dat'),
    ('sub-01', 'sub-01_acq-03_svs_slaser_cows7_metab_PFL.dat'),
    ('sub-01', 'notes.txt'),
]
METAB = ['sub-01_acq-03_cows7_metab_PFL', 'sub-01_acq-06_vapor7_metab_OCC',
         'sub-01_acq-13_cows7_metab_PAR', 'sub-10_acq-01_vapor7_metab_PFL']
MM = ['sub-01_acq-14_cows7_mm_PAR', 'sub-10_acq-02_vapor7_mm_PFL']


#*************#
#   helpers   #
#*************#
@pytest.fixture
def study(tmp_path):
    """The stand-in study on disk; the .dat files are empty, nothing reads them."""
    for subject, name in FILES:
        source = tmp_path / subject / 'mrs' / 'sourcedata'
        source.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(b'')
    return tmp_path


def _synthetic_twix(path, remove_oversampling=True, n_water=1):
    """
    Stand-in for "read_twix": tiny NIfTIs whose values are a function of the
    file name, so any process produces the same ones.
    """
    seed = zlib.crc32(os.path.basename(str(path)).encode())
    rng = np.random.default_rng(seed)
    points = 64 if remove_oversampling else 128
    dwell = 1.0 / 4000.0 if remove_oversampling else 1.0 / 8000.0
    shape = (1, 1, 1, points, 2, 4)                           # (…, T, coils, transients)

    array = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    data = gen_nifti_mrs(array.astype(np.complex64), dwell, 123.26,
                         dim_tags=['DIM_COIL', 'DIM_DYN', None])
    data.add_hdr_field('EchoTime', 0.026)
    data.add_hdr_field('RepetitionTime', 2.0)

    padded = np.zeros(shape, dtype=np.complex64)              # zero-padded like the scanner's
    padded[..., :n_water] = (rng.standard_normal(shape[:-1] + (n_water,))
                             + 1j * rng.standard_normal(shape[:-1] + (n_water,)))
    water = gen_nifti_mrs(padded, dwell, 123.26, dim_tags=['DIM_COIL', 'DIM_DYN', None])
    water.add_hdr_field('EchoTime', 0.026)
    return data, cows._acquired_water(water)


@pytest.fixture
def synthetic(monkeypatch):
    """Route "read_twix" to the stand-in and count the calls."""
    calls = []

    def fake(path, remove_oversampling=True):
        calls.append(os.path.basename(str(path)))
        return _synthetic_twix(path, remove_oversampling)

    monkeypatch.setattr(cows, 'read_twix', fake)
    return calls


def _array(nifti):
    return np.asarray(nifti[:]).reshape(nifti.shape)


def _write_mat(path, fid, sf=123.26, sw_h=4000):
    """An INSPECTOR-style "exptDat" struct."""
    import scipy.io

    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), {'exptDat': {'fid': fid[:, None], 'sf': sf, 'sw_h': sw_h,
                                             'nspecC': len(fid)}})


#*************#
#   naming    #
#*************#
def test_twix_names_parse_to_canonical_scans():
    scan = COWSScan.from_twix('/x/sub-10_acq-01_svs_slaser_vapor7_metab_PFC.dat')
    assert scan == COWSScan('sub-10', 1, 'vapor7', 'metab', 'PFL',
                            '/x/sub-10_acq-01_svs_slaser_vapor7_metab_PFC.dat')
    assert scan.stem == 'sub-10_acq-01_vapor7_metab_PFL'

    occipital = COWSScan.from_twix('sub-02_acq-09_svs_slaser_cows7_mm_Occipital.dat')
    assert (occipital.region, occipital.scan_type, occipital.acquisition) == ('OCC', 'mm', 9)
    assert COWSScan.from_twix('sub-02_acq-09_svs_slaser_cows7_mm_Thalamus.dat') is None
    assert COWSScan.from_twix('notes.txt') is None


def test_mat_names_take_region_from_the_directory_and_acq_from_the_protocol():
    scan = COWSScan.from_mat('/d/sub-02_mat/OCCIPITAL/sub-02_COWS7_MM.mat')
    assert (scan.subject, scan.region, scan.water_suppression, scan.scan_type) == \
        ('sub-02', 'OCC', 'cows7', 'mm')
    assert scan.acquisition == PROTOCOL[('OCC', 'cows7', 'mm')] == 9

    assert is_mat_water('/d/sub-02_mat/PFL/sub-02_VAPOR7_Metab_Water.mat')
    assert not is_mat_water('/d/sub-02_mat/PFL/sub-02_VAPOR7_Metab.mat')
    assert COWSScan.from_mat('/d/sub-02_mat/PFL/readme.mat') is None


def test_the_protocol_is_fifteen_distinct_acquisitions():
    assert sorted(PROTOCOL.values()) == list(range(1, 16))
    assert ('PFL', 'cows12', 'mm') not in PROTOCOL           # COWS12 has no MM partner


#*************#
#   listing   #
#*************#
def test_scans_come_sorted_by_subject_then_acquisition(study):
    stems = [s.stem for s in COWSDataModule(study).twix_scans()]
    assert stems == sorted(METAB + MM)
    assert stems == ['sub-01_acq-03_cows7_metab_PFL', 'sub-01_acq-06_vapor7_metab_OCC',
                     'sub-01_acq-13_cows7_metab_PAR', 'sub-01_acq-14_cows7_mm_PAR',
                     'sub-10_acq-01_vapor7_metab_PFL', 'sub-10_acq-02_vapor7_mm_PFL']


def test_filters_accept_every_spelling_the_study_uses(study):
    prefrontal = [s.stem for s in COWSDataModule(study, location='PFC').twix_scans()]
    assert prefrontal == [s.stem for s in COWSDataModule(study, location='pfl').twix_scans()]
    assert prefrontal == ['sub-01_acq-03_cows7_metab_PFL', 'sub-10_acq-01_vapor7_metab_PFL',
                          'sub-10_acq-02_vapor7_mm_PFL']

    vapor = COWSDataModule(study, water_sup='VAPOR')
    assert vapor.water_sup == ('vapor7',)
    assert [s.stem for s in vapor.twix_scans()] == [
        'sub-01_acq-06_vapor7_metab_OCC', 'sub-10_acq-01_vapor7_metab_PFL',
        'sub-10_acq-02_vapor7_mm_PFL']

    both = COWSDataModule(study, location=['Occipital', 'PARIETAL'], subjects='sub-01')
    assert both.regions == ('OCC', 'PAR')
    assert [s.acquisition for s in both.twix_scans()] == [6, 13, 14]


@pytest.mark.parametrize("bad", [dict(location='Thalamus'), dict(water_sup='cows9')])
def test_unknown_filters_are_refused(study, bad):
    with pytest.raises(ValueError):
        COWSDataModule(study, **bad)


#*******************#
#   header fields   #
#*******************#
def test_header_fields_read_back_as_plain_values():
    scan = COWSScan.from_twix('sub-03_acq-08_svs_slaser_cows7_metab_Occipital.dat')
    nifti, _ = _synthetic_twix(scan.path)
    cows._stamp(nifti, scan)

    # nifti_mrs wraps user-defined fields; scan_info unwraps them
    assert nifti.hdr_ext['SubjectID'] == {'Value': 'sub-03',
                                          'Description': HEADER_FIELDS['SubjectID']}
    assert scan_info(nifti) == {'SubjectID': 'sub-03', 'Region': 'OCC',
                                'WaterSuppression': 'cows7', 'ScanType': 'metab',
                                'Acquisition': 8}
    assert scan_info(gen_nifti_mrs(np.zeros((1, 1, 1, 8), complex), 1e-3, 123.0)) == {}


def test_water_keeps_the_acquired_transients_and_squeezes_a_single_one():
    _, one = _synthetic_twix('a.dat', n_water=1)
    assert one.shape == (1, 1, 1, 64, 2) and one.dim_tags == ['DIM_COIL', None, None]

    _, three = _synthetic_twix('a.dat', n_water=3)
    assert three.shape == (1, 1, 1, 64, 2, 3)
    assert three.dim_tags == ['DIM_COIL', 'DIM_DYN', None]
    assert np.abs(_array(three)).min(axis=(0, 1, 2, 3, 4)).all()     # nothing zero kept

    empty = gen_nifti_mrs(np.zeros((1, 1, 1, 64, 2, 4), np.complex64), 1 / 4000, 123.26,
                          dim_tags=['DIM_COIL', 'DIM_DYN', None])
    with pytest.raises(ValueError, match="non-zero"):
        cows._acquired_water(empty)


#*************#
#   loading   #
#*************#
def test_names_line_up_with_the_metabolite_and_mm_lists(study, synthetic):
    loader = COWSDataModule(study)
    data, water, mm, mm_water, names = loader.load_twix()

    assert names == METAB and loader.mm_names == MM
    assert [s.stem for s in loader.scans] == METAB and [s.stem for s in loader.mm_scans] == MM
    assert len(data) == len(water) == 4 and len(mm) == len(mm_water) == 2
    for nifti, ref, name in zip(data + mm, water + mm_water, names + loader.mm_names):
        info = scan_info(nifti)
        assert scan_info(ref) == info
        assert name == (f"{info['SubjectID']}_acq-{info['Acquisition']:02d}"
                        f"_{info['WaterSuppression']}_{info['ScanType']}_{info['Region']}")
        assert nifti.dim_tags == ['DIM_COIL', 'DIM_DYN', None]
        assert ref.dim_tags == ['DIM_COIL', None, None]
        assert nifti.hdr_ext['EchoTime'] == 0.026
    assert all(scan_info(n)['Region'] == 'PFL' for n in data + mm
               if scan_info(n)['SubjectID'] == 'sub-10')
    assert sorted(synthetic) == sorted(name for _, name in FILES if name.endswith('.dat'))


def test_failures_raise_by_default_and_are_collected_otherwise(study, monkeypatch):
    def fake(path, remove_oversampling=True):
        if 'acq-06' in str(path):
            raise IndexError("index 0 is out of bounds for axis 0 with size 0")
        return _synthetic_twix(path, remove_oversampling)

    monkeypatch.setattr(cows, 'read_twix', fake)

    with pytest.raises(RuntimeError, match="sub-01_acq-06_svs_slaser_vapor7_metab_Occipital"):
        COWSDataModule(study).load_twix()

    loader = COWSDataModule(study, strict=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        data, _, mm, _, names = loader.load_twix()
    assert names == [n for n in METAB if 'acq-06' not in n] and loader.mm_names == MM
    assert len(loader.load_failures) == 1 and 'acq-06' in loader.load_failures[0][0]
    assert any('acq-06' in str(w.message) for w in caught)


def test_cache_round_trips_the_scans_and_skips_the_twix_read(study, synthetic, tmp_path):
    cache = tmp_path / 'cache'
    loader = COWSDataModule(study, cache_dir=cache)
    first = loader.load_twix()
    assert len(synthetic) == 6
    assert sorted(os.listdir(cache)) == sorted(
        ['index.json'] + [f"{s}.nii" for s in METAB + MM]
        + [f"{s}_water.nii" for s in METAB + MM])
    with open(cache / 'index.json') as handle:
        index = json.load(handle)
    assert index['sub-10_acq-01_vapor7_metab_PFL.nii']['region'] == 'PFL'

    second = COWSDataModule(study, cache_dir=cache).load_twix()
    assert len(synthetic) == 6, "the second load read the TWIX files again"
    assert second[4] == first[4] == METAB
    for lists in zip(first[:4], second[:4]):
        for a, b in zip(*lists):
            assert np.array_equal(_array(a), _array(b))
            assert a.dim_tags == b.dim_tags and a.dwelltime == b.dwelltime
            assert scan_info(a) == scan_info(b) and b.hdr_ext['EchoTime'] == 0.026


def test_cache_is_bypassed_for_other_oversampling_or_a_changed_source(study, synthetic, tmp_path):
    cache = tmp_path / 'cache'
    COWSDataModule(study, cache_dir=cache, subjects='sub-10').load_twix()
    assert len(synthetic) == 2

    # oversampled data is a different array and gets its own files
    kept = COWSDataModule(study, cache_dir=cache, subjects='sub-10',
                          remove_oversampling=False).load_twix()
    assert len(synthetic) == 4 and kept[0][0].shape[3] == 128
    assert (cache / 'sub-10_acq-01_vapor7_metab_PFL_os.nii').is_file()

    # a source file that changed since is read again, not served stale
    (study / 'sub-10' / 'mrs' / 'sourcedata'
     / 'sub-10_acq-01_svs_slaser_vapor7_metab_PFC.dat').write_bytes(b'x')
    COWSDataModule(study, cache_dir=cache, subjects='sub-10').load_twix()
    assert len(synthetic) == 5


def test_workers_deliver_the_same_objects_as_the_serial_path(study, synthetic):
    if multiprocessing.get_start_method() != 'fork':
        pytest.skip("the stand-in reader only reaches forked workers")

    serial = COWSDataModule(study).load_twix()
    parallel = COWSDataModule(study, workers=2).load_twix()

    assert parallel[4] == serial[4]
    for lists in zip(serial[:4], parallel[:4]):
        for a, b in zip(*lists):
            assert np.array_equal(_array(a), _array(b))
            assert a.dim_tags == b.dim_tags and scan_info(a) == scan_info(b)
            assert b.hdr_ext['EchoTime'] == 0.026


#*********#
#   mat   #
#*********#
def test_mat_fids_land_where_inspector_put_them(tmp_path):
    """
    INSPECTOR stores the FID in the FSL-MRS user-facing convention, so a
    line at -326 Hz (NAA's offset from water at 123.26 MHz) must come out at
    4.65 - 326 / 123.26 = 2.0 ppm on the FSL axis, not mirrored to 7.3.
    """
    from fsl_mrs.core import MRS

    t = np.arange(2048) / 4000.0
    fid = np.exp(2j * np.pi * -326.0 * t) * np.exp(-t / 0.05)
    path = tmp_path / 'sub-02_mat' / 'OCCIPITAL' / 'sub-02_VAPOR7_Metab.mat'
    _write_mat(path, fid)

    nifti = read_mat(path)
    assert nifti.shape == (1, 1, 1, 2048) and nifti.dwelltime == pytest.approx(1 / 4000)
    assert nifti.hdr_ext['EchoTime'] == 0.026 and nifti.hdr_ext['RepetitionTime'] == 2.0

    mrs = MRS(FID=_array(nifti)[0, 0, 0], cf=123.26, bw=4000.0, nucleus='1H')
    ppm, spec = mrs.getAxes(ppmlim=(0, 10)), np.abs(mrs.get_spec(ppmlim=(0, 10)))
    assert ppm[np.argmax(spec)] == pytest.approx(2.005, abs=0.01)


def test_load_mats_pairs_each_scan_with_its_water(tmp_path):
    root = tmp_path / 'derivatives' / 'mrs_mat' / 'sub-02_mat'
    fid = np.exp(-np.arange(2048) / 400.0) + 0j
    _write_mat(root / 'OCCIPITAL' / 'sub-02_VAPOR7_Metab.mat', fid)
    _write_mat(root / 'OCCIPITAL' / 'sub-02_VAPOR7_Metab_Water.mat', 3 * fid)
    _write_mat(root / 'PFL' / 'sub-02_COWS7_MM.mat', fid)             # no water partner

    loader = COWSDataModule(tmp_path, location=['OCC', 'PFC'])
    data, water, mm, mm_water, names = loader.load_mats()

    assert names == ['sub-02_acq-06_vapor7_metab_OCC'] and len(water) == 1
    assert loader.mm_names == ['sub-02_acq-04_cows7_mm_PFL'] and mm_water == [None]
    assert scan_info(water[0]) == scan_info(data[0])
    assert np.allclose(_array(water[0]), 3 * _array(data[0]))


#***************#
#   factory     #
#***************#
def test_the_factory_splits_by_subject(study, synthetic, monkeypatch):
    received = {}

    class Recorder(cows.Augmentrum):
        """Records the call instead of building; keeps the class-level queries."""
        def __init__(self, data, water=None, groups=None, **kwargs):
            received.update(groups=groups, n=len(data), kwargs=kwargs)

    monkeypatch.setattr(cows, 'Augmentrum', Recorder)
    cows.COWSData(study)
    assert received['n'] == 4
    assert received['groups'] == ['sub-01', 'sub-01', 'sub-01', 'sub-10']
    # the default pipelines draw coils and transients, so their ranges go through
    assert received['kwargs']['n_coils'] == (1, None)
    assert received['kwargs']['n_averages'] == (1, None)

    received.clear()
    cows.COWSData(study, subjects='sub-10', pipelines={})
    assert received['n'] == 1 and received['groups'] == ['sub-10']
    # nothing in an empty pipeline takes a sampling range, so none is passed
    assert 'n_coils' not in received['kwargs'] and 'n_averages' not in received['kwargs']


def test_the_factory_builds_an_augmentrum(study, synthetic):
    aug = cows.COWSData(study, backend='numpy', volatile=True, pipelines={}, batch_size=1,
                        val_frac=0.25, test_frac=0.25)
    assert set(aug.splits) == {'train', 'val', 'test'}
    assert sum(len(data.list()) for data, _ in aug.splits.values()) == 4


#****************#
#   real data    #
#****************#
@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_DIR, 'sub-02', 'mrs', 'sourcedata')),
                    reason="COWS data not present")
def test_real_twix_arrives_standard_with_naa_at_two_ppm():
    """
    One real scan through spec2nii and RawProcessor(conj=False): the
    acquisition as the protocol specifies it, the water with its single
    transient, spec2nii's header intact, and the spectrum the right way round.
    """
    from fsl_mrs.core import MRS
    from augmentrum.core import Backend, NIfTI_MRS_Plus
    from augmentrum.processing.raw_processing import RawProcessor

    loader = COWSDataModule(DATA_DIR, subjects='sub-02', location='OCC', water_sup='vapor7')
    data, water, mm, mm_water, names = loader.load_twix()
    assert names == ['sub-02_acq-06_vapor7_metab_OCC']
    assert loader.mm_names == ['sub-02_acq-07_vapor7_mm_OCC']

    met, ref = data[0], water[0]
    assert met.shape == (1, 1, 1, 2048, 32, 32) and ref.shape == (1, 1, 1, 2048, 32)
    assert met.dwelltime == pytest.approx(2.5e-4)
    assert met.spectrometer_frequency[0] == pytest.approx(123.26, abs=0.01)
    assert met.hdr_ext['EchoTime'] == 0.026 and met.hdr_ext['RepetitionTime'] == 2.0
    assert 'TxOffset' in met.hdr_ext and scan_info(ref) == scan_info(met)

    out, _ = RawProcessor(conj=False)(
        NIfTI_MRS_Plus([met], backend=Backend.NIFTI_LIST, volatile=True),
        NIfTI_MRS_Plus([ref], backend=Backend.NIFTI_LIST, volatile=True))
    processed = out.list()[0]
    mrs = MRS(FID=_array(processed)[0, 0, 0], cf=float(processed.spectrometer_frequency[0]),
              bw=1.0 / float(processed.dwelltime), nucleus='1H')
    ppm, spec = mrs.getAxes(ppmlim=(1.5, 3.5)), np.real(mrs.get_spec(ppmlim=(1.5, 3.5)))
    naa = (ppm > 1.8) & (ppm < 2.2)
    cr = (ppm > 2.8) & (ppm < 3.2)
    assert ppm[naa][np.argmax(spec[naa])] == pytest.approx(2.01, abs=0.02)
    assert ppm[cr][np.argmax(spec[cr])] == pytest.approx(3.03, abs=0.03)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(os.path.join(
    DATA_DIR, 'sub-01', 'mrs', 'sourcedata',
    'sub-01_acq-06_svs_slaser_vapor7_metab_Occipital.dat')), reason="COWS data not present")
def test_the_truncated_upstream_file_fails_cleanly():
    loader = COWSDataModule(DATA_DIR, subjects='sub-01', location='Occipital',
                            water_sup='vapor7')
    with pytest.raises(RuntimeError, match="sub-01_acq-06"):
        loader.load_twix()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data, _, mm, _, _ = loader.load_twix(strict=False)
    assert data == [] and loader.mm_names == ['sub-01_acq-07_vapor7_mm_OCC']
    assert len(loader.load_failures) == 1


def test_a_compressed_cache_is_written_on_request_and_read_either_way(study, synthetic, tmp_path,
                                                                     monkeypatch):
    cache = tmp_path / 'cache'
    first = COWSDataModule(study, cache_dir=cache, subjects='sub-10',
                           compress_cache=True).load_twix()
    assert (cache / 'sub-10_acq-01_vapor7_metab_PFL.nii.gz').is_file()
    assert not (cache / 'sub-10_acq-01_vapor7_metab_PFL.nii').exists()

    # the default (uncompressed) loader reads the gzipped cache instead of the TWIX
    monkeypatch.setattr(cows, 'read_twix', lambda *a, **k: pytest.fail("TWIX read"))
    monkeypatch.setattr(cows, '_load_twix_scan', lambda *a, **k: pytest.fail("TWIX read"))
    second = COWSDataModule(study, cache_dir=cache, subjects='sub-10').load_twix()
    assert len(second[0]) == len(first[0])
    for a, b in zip(first[0], second[0]):
        assert np.array_equal(a[:], b[:])
