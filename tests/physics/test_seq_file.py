####################################################################################################
#                                      test_seq_file.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: Checks augmentrum.physics.seq_file against a real Pulseq ".seq" file, built at test      #
#          time via pypulseq (skipped entirely if pypulseq is not installed).                       #
#                                                                                                  #
####################################################################################################

"""
Tests for augmentrum.physics.seq_file.
"""

#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from augmentrum.physics.seq_file import GAMMA_HZ_PER_T
from tests.physics._seq_fixtures import write_synthetic_seq

pp = pytest.importorskip('pypulseq')


@pytest.fixture
def seq_path(tmp_path):
    path = str(tmp_path / 'synthetic.seq')
    write_synthetic_seq(path, n_samples=32, fov_m=0.01, b0_tesla=3.0)
    return path


#**************************************************************************************************#
#                                       Class TestLoadSeqFile                                       #
#**************************************************************************************************#
class TestLoadSeqFile:
    """Shapes, units and definitions read off a real .seq file."""

    def test_definitions_are_read_verbatim(self, seq_path):
        from augmentrum.physics.seq_file import load_seq_file

        out = load_seq_file(seq_path)
        assert float(out.definitions['B0']) == pytest.approx(3.0)
        np.testing.assert_allclose(np.asarray(out.definitions['FOV']), [0.01, 0.01, 0.01])

    def test_gradient_raster_is_uniform_and_matches_grad_raster_time(self, seq_path):
        from augmentrum.physics.seq_file import load_seq_file

        out = load_seq_file(seq_path)
        assert out.gradients_t_per_m.shape == (3, out.t_grid.shape[0])
        diffs = np.diff(out.t_grid)
        np.testing.assert_allclose(diffs, out.dt)

    def test_k_traj_has_one_column_per_adc_sample(self, seq_path):
        from augmentrum.physics.seq_file import load_seq_file

        out = load_seq_file(seq_path)
        assert out.k_traj_m.shape == (3, 32)
        assert out.t_adc.shape == (32,)

    def test_gradient_amplitude_converts_hz_per_m_to_t_per_m(self, seq_path):
        """pypulseq's own convention is Hz/m; dividing by gamma must land in
        a physically sane T/m range (clinical gradients are well under 1 T/m)."""
        from augmentrum.physics.seq_file import load_seq_file

        out = load_seq_file(seq_path)
        peak_t_per_m = np.max(np.abs(out.gradients_t_per_m))
        assert 0.0 < peak_t_per_m < 1.0

    def test_nonexistent_file_raises(self, tmp_path):
        from augmentrum.physics.seq_file import load_seq_file

        with pytest.raises(FileNotFoundError):
            load_seq_file(str(tmp_path / 'does_not_exist.seq'))

    def test_explicit_default_gamma_matches_omitted_gamma(self, seq_path):
        from augmentrum.physics.seq_file import load_seq_file

        out = load_seq_file(seq_path, gamma_hz_per_t=GAMMA_HZ_PER_T)
        default_out = load_seq_file(seq_path)
        np.testing.assert_allclose(out.gradients_t_per_m, default_out.gradients_t_per_m)
