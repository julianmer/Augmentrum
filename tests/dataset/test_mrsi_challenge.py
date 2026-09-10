####################################################################################################
#                                     test_mrsi_challenge.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-10                                                                              #
#                                                                                                  #
# Purpose: Holds the MRSI Challenge loader to fetching exactly what a run asks for, laying it out  #
#          the way the Zenodo release does, and carving splits that can never overlap.             #
#                                                                                                  #
####################################################################################################

"""
Tests for the MRSI Challenge loader.

None of these reach the network or need the release. The download itself is
covered in tests/utils/test_download.py; what is guarded here is what this
dataset adds on top - which subjects a splits spec selects, that only those
are fetched, where they land, and that a tiny stand-in release goes through
the whole factory.
"""

#*************#
#   imports   #
#*************#
import os
import zipfile

import numpy as np
import pytest

from augmentrum.dataset.mrsi_challenge import MRSIChallengeDataModule


#*************#
#   helpers   #
#*************#
def _stand_in_zip(tmp_path, subject, payload=b'not really HDF5'):
    """A zip laid out like the release's, holding a placeholder .mat."""
    archive = tmp_path / 'served' / f'{subject}.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr(f'{subject}/{subject}_all.mat', payload)
        if subject.startswith('Test'):
            bundle.writestr(f'{subject}/{subject}_all_truth.mat', payload)
    return archive


@pytest.fixture
def served(tmp_path, monkeypatch):
    """
    Point the loader at zips on disk instead of Zenodo.

    Returns the list of subjects that were "downloaded", in order.
    """
    from augmentrum.utils import download

    fetched = []

    def fake_fetch(url, target, md5=None, size=None, progress=True, **kwargs):
        subject = os.path.basename(url)
        fetched.append(subject)
        target.write_bytes(_stand_in_zip(tmp_path, subject).read_bytes())
        return target

    monkeypatch.setattr(download, 'fetch', fake_fetch)
    monkeypatch.setattr(
        MRSIChallengeDataModule, '_record_files',
        classmethod(lambda cls: {f'{s}.zip': (s, 1, 'md5') for s in cls.ALL_SUBJECTS}),
    )
    return fetched


#**************#
#   resolve    #
#**************#
def test_default_splits_are_the_whole_release():
    splits = MRSIChallengeDataModule.resolve()
    assert splits['train'] == MRSIChallengeDataModule.TRAIN_SUBJECTS[:19]
    assert splits['val'] == MRSIChallengeDataModule.TRAIN_SUBJECTS[19:]
    assert splits['test_track1'] == MRSIChallengeDataModule.TRACK1_SUBJECTS
    assert splits['test_track2'] == MRSIChallengeDataModule.TRACK2_SUBJECTS


def test_counts_take_train_before_val_and_leave_out_the_rest():
    splits = MRSIChallengeDataModule.resolve({'train': 6, 'val': 2, 'test_track1': 0})
    assert splits == {'train': tuple(f'Sub{i}' for i in range(1, 7)),
                      'val': ('Sub7', 'Sub8')}


def test_explicit_subjects_are_removed_from_the_pool():
    splits = MRSIChallengeDataModule.resolve({'train': ('Sub1', 'Sub7'), 'val': 2,
                                              'test_track1': ('TestSub3',)})
    assert splits == {'train': ('Sub1', 'Sub7'), 'val': ('Sub2', 'Sub3'),
                      'test_track1': ('TestSub3',)}


@pytest.mark.parametrize("bad", [
    {'train': 25},                                   # more than there are
    {'train': ('Sub1',), 'val': ('Sub1',)},          # overlap
    {'train': ('TestSub1',)},                        # a test subject in training
    {'test': 3},                                     # no such split
])
def test_impossible_splits_are_refused(bad):
    with pytest.raises(ValueError):
        MRSIChallengeDataModule.resolve(bad)


#**************#
#   fetching   #
#**************#
def test_fetch_unpacks_the_release_layout_and_drops_the_zip(tmp_path, served):
    root = tmp_path / 'release'
    MRSIChallengeDataModule.fetch(['Sub2', 'TestSub10'], root, progress=False)

    assert (root / 'Sub2' / 'Sub2_all.mat').is_file()
    assert (root / 'TestSub10' / 'TestSub10_all.mat').is_file()
    assert (root / 'TestSub10' / 'TestSub10_all_truth.mat').is_file()
    assert not list(root.glob('*.zip')), "the archive was kept after unpacking"
    assert served == ['Sub2', 'TestSub10']


def test_only_missing_subjects_are_fetched(tmp_path, served):
    root = tmp_path / 'release'
    MRSIChallengeDataModule.fetch(['Sub1'], root, progress=False)
    MRSIChallengeDataModule.fetch(['Sub1', 'Sub2'], root, progress=False)
    assert served == ['Sub1', 'Sub2']


def test_unknown_subjects_are_refused_before_anything_is_fetched(tmp_path, served):
    with pytest.raises(ValueError, match='Sub99'):
        MRSIChallengeDataModule.fetch(['Sub1', 'Sub99'], tmp_path / 'release')
    assert served == []


def test_load_without_download_names_what_to_fetch(tmp_path):
    module = MRSIChallengeDataModule(tmp_path / 'release', download=False)
    with pytest.raises(FileNotFoundError, match=r"fetch\(\['Sub3'\]"):
        module.load(['Sub3'])


#*****************#
#   path layout   #
#*****************#
def test_paths_follow_the_release_layout(tmp_path):
    module = MRSIChallengeDataModule(tmp_path, signal='clean')
    assert module.mat_path('Sub4', need_truth=True) == \
        str(tmp_path / 'Sub4' / 'Sub4_all.mat')
    assert module.mat_path('TestSub2', need_truth=False) == \
        str(tmp_path / 'TestSub2' / 'TestSub2_all.mat')
    assert module.mat_path('TestSub2', need_truth=True) == \
        str(tmp_path / 'TestSub2' / 'TestSub2_all_truth.mat')
    assert module.nifti_paths('Sub4') is None, "no NIfTI on disk means no NIfTI path"


#*************#
#   signals   #
#*************#
@pytest.mark.parametrize("signal, terms", [
    ('clean',          ((1, 'meta'),)),
    ('nuisance_free',  ((1, 'all'), (-1, 'nuisance'))),
    ('macromolecules', ((1, 'mm'),)),
    ('meta+mm',        ((1, 'meta'), (1, 'mm'))),
    ('meta + mm + baseline', ((1, 'meta'), (1, 'mm'), (1, 'baseline'))),
    ('all-nuisance-mm', ((1, 'all'), (-1, 'nuisance'), (-1, 'mm'))),
])
def test_signals_expand_to_signed_components(signal, terms):
    assert MRSIChallengeDataModule.parse_signal(signal) == terms


@pytest.mark.parametrize("bad", ['xtMeta', 'meta+', 'meta+water', '', 'meta*mm'])
def test_unknown_signals_are_refused(bad):
    with pytest.raises(ValueError):
        MRSIChallengeDataModule.parse_signal(bad)


def test_a_preset_and_its_expression_share_one_cache_file(tmp_path):
    module = MRSIChallengeDataModule(tmp_path)
    assert module._cache_path('Sub1', 'clean') == module._cache_path('Sub1', 'meta')
    assert module._cache_path('TestSub1', 'nuisance_free').endswith('TestSub1_all-nuisance.nii')


#*****************#
#   orientation   #
#*****************#
def test_mat_arrays_land_on_the_nifti_grid():
    """h5py hands the .mat over as (Z, X, Y); the NIfTI grid is (X, Y, Z), mirrored in Y and Z."""
    arr = np.zeros((2, 3, 4))
    arr[0, 1, 2] = 1                                     # z=0, x=1, y=2
    out = MRSIChallengeDataModule._to_nifti_order(arr)
    assert out.shape == (3, 4, 2)
    assert out[1, 4 - 1 - 2, 2 - 1 - 0] == 1

    spectral = np.zeros((5, 2, 3, 4))                    # (T, Z, X, Y)
    assert MRSIChallengeDataModule._to_nifti_order(spectral).shape == (3, 4, 2, 5)


#*************#
#   factory   #
#*************#
h5py = pytest.importorskip("h5py")


def _write_mat(path, components=('xtMeta', 'xtAll', 'xtNuisance'), shape=(8, 4, 6, 5)):
    """A miniature .mat: complex compound arrays as h5py sees them, (T, Z, X, Y)."""
    t, z, x, y = shape
    rng = np.random.default_rng(0)
    compound = np.dtype([('real', '<f8'), ('imag', '<f8')])

    def field():
        arr = np.empty((t, z, x, y), dtype=compound)
        arr['real'] = rng.standard_normal((t, z, x, y))
        arr['imag'] = rng.standard_normal((t, z, x, y))
        return arr

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        for name in components:
            f[name] = field()
        f['brainMask'] = np.ones((z, x, y))
        f['hzpppm'] = np.array([[127.732434]])
        f['ppmoff'] = np.array([[4.65]])
        f['t'] = (1.66e-3 + np.arange(t) * 0.83e-3)[:, None]


def test_factory_builds_only_the_requested_splits(tmp_path):
    pytest.importorskip("fsl_mrs")
    from augmentrum.dataset.mrsi_challenge import MRSIChallengeData

    root = tmp_path / 'release'
    for subject in ('Sub1', 'Sub2', 'Sub3'):
        _write_mat(root / subject / f'{subject}_all.mat')

    aug = MRSIChallengeData(root, splits={'train': 2, 'val': 1}, batch_size=1,
                            download=False, with_aux=True, pipelines={})

    assert set(aug.splits) == {'train', 'val'}
    assert aug.subject_names == {'train': ['Sub1', 'Sub2'], 'val': ['Sub3']}
    assert aug.aux['val'][0]['brainMask'].shape == (6, 5, 4)         # (X, Y, Z)
    assert aug.data_module.acquisition['spectrometer_frequency_mhz'] == 127.732434


def test_each_split_can_load_its_own_signal(tmp_path):
    """Train on the noiseless metabolites, evaluate track 1 against metabolites + MM."""
    pytest.importorskip("fsl_mrs")
    from augmentrum.dataset.mrsi_challenge import MRSIChallengeData

    root = tmp_path / 'release'
    _write_mat(root / 'Sub1' / 'Sub1_all.mat')
    _write_mat(root / 'TestSub1' / 'TestSub1_all.mat', components=('xtAll',))
    _write_mat(root / 'TestSub1' / 'TestSub1_all_truth.mat',
               components=('xtMeta', 'xtAll', 'xtNuisance', 'xtMM', 'xtBaseline'))

    aug = MRSIChallengeData(root, splits={'train': 1, 'test_track1': 1}, batch_size=1,
                            signal={'test_track1': 'meta+mm'}, download=False, pipelines={})
    assert aug.signals == {'train': 'clean', 'test_track1': 'meta+mm'}

    module = aug.data_module
    with h5py.File(root / 'TestSub1' / 'TestSub1_all_truth.mat') as f:
        meta, mm = (f[k][()] for k in ('xtMeta', 'xtMM'))
    expected = module._to_nifti_order((meta['real'] + mm['real']) + 1j * (meta['imag'] + mm['imag']))
    np.testing.assert_allclose(module.read_component('TestSub1', 'meta+mm'), expected, rtol=1e-6)


def test_a_component_the_subject_lacks_is_an_error_not_a_substitute(tmp_path):
    """Track 2 has no nuisance; asking for 'all-nuisance' there must fail loudly."""
    root = tmp_path / 'release'
    _write_mat(root / 'TestSub10' / 'TestSub10_all_truth.mat',
               components=('xtMeta', 'xtAll', 'xtMM', 'xtBaseline'))
    module = MRSIChallengeDataModule(root, download=False)
    with pytest.raises(KeyError, match="nuisance"):
        module.read_component('TestSub10', 'nuisance_free')
    _write_mat(root / 'Sub1' / 'Sub1_all.mat')                  # training: no xtMM
    with pytest.raises(KeyError, match="mm"):
        module.read_component('Sub1', 'meta+mm')
