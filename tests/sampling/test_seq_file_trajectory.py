####################################################################################################
#                                 test_seq_file_trajectory.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: The "seq_file" trajectory - a real Pulseq ".seq" file used as the k-space trajectory     #
#          (and gradient waveform) source for KspaceUndersampling, instead of an analytically       #
#          generated one. Skipped entirely if pypulseq is not installed.                            #
#                                                                                                  #
####################################################################################################

"""
Tests for augmentrum.sampling.kspace_sampling.SeqFile.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.sampling.kspace_sampling import KspaceUndersampling, TrajectoryRegistry
from tests.physics._seq_fixtures import write_synthetic_seq

pytest.importorskip('pypulseq')

HEADER = {
    'dim': [4, 16, 16, 1, 1, 1, 1, 1],
    'pixdim': [1.0, 1.25, 1.25, 1.25, 1.0, 1.0, 1.0, 1.0],
    'DwellTime': 1e-4,
    'SpectrometerFrequency': [127.74],
}


@pytest.fixture
def single_shot_seq(tmp_path):
    path = str(tmp_path / 'single_shot.seq')
    write_synthetic_seq(path, n_samples=32, fov_m=0.02, b0_tesla=3.0, n_shots=1)
    return path


@pytest.fixture
def multi_shot_seq(tmp_path):
    path = str(tmp_path / 'multi_shot.seq')
    write_synthetic_seq(path, n_samples=32, fov_m=0.02, b0_tesla=3.0, n_shots=4)
    return path


#**************************************************************************************************#
#                                    Class TestTrajectoryGeneration                                 #
#**************************************************************************************************#
class TestTrajectoryGeneration:
    """TrajectoryRegistry.generate('seq_file', ...) against a real .seq file."""

    def test_registered_under_seq_file(self):
        assert 'seq_file' in TrajectoryRegistry.available()

    def test_missing_seq_file_param_raises(self):
        with pytest.raises(ValueError, match="seq_file"):
            TrajectoryRegistry.generate('seq_file', HEADER, {}, like=None)

    def test_shots_are_in_cycles_per_m_not_normalized(self, single_shot_seq):
        """Coordinates must be on the same physical (cycles/m) scale every
        other trajectory in this module returns - not normalized to
        [-0.5, 0.5) or any other convention."""
        shots, meta = TrajectoryRegistry.generate(
            'seq_file', HEADER, {'seq_file': single_shot_seq}, like=None)
        kmax = meta['kmax'][0]
        peak = max(float(np.max(np.abs(s))) for s in shots)
        # A real trajectory should reach roughly kmax, not some other scale
        # (e.g. 1.0 for a normalized convention, or raw Hz/m).
        assert 0.1 * kmax < peak < 10.0 * kmax

    def test_single_shot_is_detected_as_one_shot(self, single_shot_seq):
        shots, meta = TrajectoryRegistry.generate(
            'seq_file', HEADER, {'seq_file': single_shot_seq}, like=None)
        assert meta['n_shots'] == 1
        assert len(shots) == 1
        assert shots[0].shape == (32, 2)

    def test_multi_shot_boundaries_are_detected_from_adc_gaps(self, multi_shot_seq):
        shots, meta = TrajectoryRegistry.generate(
            'seq_file', HEADER, {'seq_file': multi_shot_seq}, like=None)
        assert meta['n_shots'] == 4
        assert all(s.shape == (32, 2) for s in shots)

    def test_explicit_n_shots_overrides_gap_detection(self, multi_shot_seq):
        shots, meta = TrajectoryRegistry.generate(
            'seq_file', HEADER, {'seq_file': multi_shot_seq, 'n_shots': 2}, like=None)
        assert meta['n_shots'] == 2
        assert len(shots) == 2

    def test_meta_carries_the_gradient_waveform(self, multi_shot_seq):
        """The whole point: the real gradient waveform must be reachable
        alongside the trajectory it was derived from, for anything besides
        undersampling that wants it."""
        _, meta = TrajectoryRegistry.generate(
            'seq_file', HEADER, {'seq_file': multi_shot_seq}, like=None)
        assert meta['gradients_t_per_m'].shape[0] == 3
        assert meta['dt'] > 0
        assert meta['seq_definitions']['B0'] == pytest.approx(3.0)

    def test_3d_geometry_keeps_all_three_axes(self, single_shot_seq):
        header_3d = dict(HEADER, dim=[4, 16, 16, 4, 1, 1, 1, 1])
        shots, meta = TrajectoryRegistry.generate(
            'seq_file', header_3d, {'seq_file': single_shot_seq}, like=None)
        assert shots[0].shape[1] == 3
        assert len(meta['kmax']) == 3


#**************************************************************************************************#
#                                Class TestThroughKspaceUndersampling                                #
#**************************************************************************************************#
class TestThroughKspaceUndersampling:
    """The actual point: driving KspaceUndersampling with a real .seq file."""

    def _volume(self):
        rng = np.random.default_rng(0)
        return (rng.standard_normal((1, 16, 16, 1, 8))
               + 1j * rng.standard_normal((1, 16, 16, 1, 8))).astype(np.complex64)

    def test_nufft_mode_runs_and_records_meta(self, multi_shot_seq):
        module = KspaceUndersampling(
            ksp_mode='nufft', trajectory='seq_file',
            traj_params={'seq_file': multi_shot_seq},
            pixdim=(1.25, 1.25, 1.25), acceleration_factor=1.0)
        x = self._volume()
        out, _ = module.process_tensor(x)

        assert out.shape == x.shape
        assert module.last_meta_['trajectory_type'] == 'seq_file'
        assert module.last_meta_['n_shots'] == 4
        assert 'gradients_t_per_m' in module.last_meta_

    def test_gridded_mode_undersamples_shots(self, multi_shot_seq):
        module = KspaceUndersampling(
            ksp_mode='gridded', trajectory='seq_file',
            traj_params={'seq_file': multi_shot_seq},
            undersampling='prefix', acceleration_factor=2.0,
            pixdim=(1.25, 1.25, 1.25))
        x = self._volume()
        out, _ = module.process_tensor(x)

        assert out.shape == x.shape
        assert module.last_meta_['n_shots_retained'] == 2

    def test_incompatible_undersampling_method_raises(self, multi_shot_seq):
        module = KspaceUndersampling(
            ksp_mode='gridded', trajectory='seq_file',
            traj_params={'seq_file': multi_shot_seq},
            undersampling='keep_acs', acceleration_factor=2.0,
            pixdim=(1.25, 1.25, 1.25))
        with pytest.raises(ValueError, match="not compatible"):
            module.process_tensor(self._volume())
