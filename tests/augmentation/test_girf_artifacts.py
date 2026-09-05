####################################################################################################
#                                  test_girf_artifacts.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-05                                                                              #
#                                                                                                  #
# Purpose: GIRFArtifacts' trajectory-error, GIRF-phase and concomitant-field terms, against a       #
#          tiny synthetic .seq fixture generated at test time (via GIRF-Sim's own mrs_seq          #
#          builder, at slew-safe but deliberately small sizes so the whole suite runs in seconds).  #
#                                                                                                  #
####################################################################################################

"""
Tests for GIRFArtifacts (GIRF trajectory/phase error and concomitant-field artifacts).

Requires GIRF-Sim (girf_module, girf_synthetic, girf_mrsi_extensions, seq_girf,
bacon_girf, mrs_seq) to be importable; skipped entirely otherwise.
"""

#*************#
#   imports   #
#*************#
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import girf_module            # noqa: F401
    import girf_mrsi_extensions    # noqa: F401
    import seq_girf                # noqa: F401
    import bacon_girf               # noqa: F401

    # "mrs_seq" (the .seq-file builder used only to generate this test's
    # fixture, never imported by GIRFArtifacts itself) is not part of
    # GIRF-Sim's own installed package surface (its pyproject.toml lists only
    # the 5 flat modules). For an editable install, girf_module.__file__
    # still points at the source checkout, which is where "mrs_seq" lives -
    # fall back to that rather than requiring it on sys.path already.
    try:
        import mrs_seq              # noqa: F401
    except ImportError:
        repo_root = str(Path(girf_module.__file__).resolve().parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import mrs_seq              # noqa: F401

    from mrs_seq.system import build_system
    from mrs_seq.localization.steam import build_steam
    GIRF_SIM_AVAILABLE = True
except ImportError:
    GIRF_SIM_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GIRF_SIM_AVAILABLE, reason="GIRF-Sim not installed")

if GIRF_SIM_AVAILABLE:
    from augmentrum.augmentation.girf_artifacts import GIRFArtifacts

MATRIX = (4, 4, 1)
B0_TESLA = 3.0


#**************#
#   fixtures   #
#**************#
@pytest.fixture(scope='module')
def tiny_seq_file(tmp_path_factory):
    """
    A slew-safe, deliberately tiny 2-shot spiral MRSI2D ".seq" file - just
    big enough to exercise every code path (multiple shots, multiple
    severity segments) while keeping the whole suite fast.
    """
    system = build_system('generic_3T', grad_raster_time=4e-5)
    seq, _ = build_steam(
        system,
        acquisition_mode='mrsi2d',
        voi_mm=(60.0, 60.0, 20.0),
        TE=0.020, TM=0.020, TR=0.2,
        fov_mm=(80.0, 80.0), matrix=(4, 4),
        n_shots=3, samples_per_shot=32,
        spectral_bandwidth_hz=2000.0, spectral_points=6,
        target_fid_duration=0.1,
        seq_name='tiny_test',
    )
    path = tmp_path_factory.mktemp('seq') / 'tiny_test.seq'
    seq.write(str(path))
    return str(path)


def _phantom(batch=1):
    """A small disc phantom, decaying FID per voxel."""
    nx, ny, nz = MATRIX
    n_t = 6
    yy, xx = np.mgrid[0:nx, 0:ny]
    disc = ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2 < (nx / 2.5) ** 2).astype(np.float32)
    t = np.arange(n_t) / 2000.0
    fid = np.exp(-8.0 * t).astype(np.complex64)
    vol = np.zeros((batch, nx, ny, nz, n_t), np.complex64)
    vol[:] = disc[None, :, :, None, None] * fid[None, None, None, None, :]
    return vol


#: matches the fixture's fov_mm=(80,80) over matrix=(4,4) - 20 mm/voxel.
#: Without this, GIRFArtifacts falls back to its isotropic-1mm default,
#: which disagrees with the .seq file's own declared geometry and makes
#: any hand-rolled reference computed from the .seq file's real FOV
#: incomparable (same NUFFT, different k-space normalization).
PIXDIM_MM = (20.0, 20.0, 20.0)


def _run(seq_file, **kwargs):
    kwargs.setdefault('b0_tesla', B0_TESLA)
    kwargs.setdefault('pixdim', PIXDIM_MM)
    girf = GIRFArtifacts(seq_file=seq_file, **kwargs)
    out, _ = girf.process_tensor(_phantom(), sw_hz=2000.0)
    return np.asarray(out), girf


#**************************************************************************************************#
#                                     Class TestIdentityWhenOff                                    #
#**************************************************************************************************#
class TestIdentityWhenOff:
    """All three terms disabled -> the plain NUFFT round trip is never run."""

    def test_all_disabled_is_bitwise_identity(self, tiny_seq_file):
        vol = _phantom()
        out, girf = _run(tiny_seq_file, include_trajectory_error=False,
                         include_girf_phase=False, include_concomitant=False)
        assert np.array_equal(out, vol)
        assert girf.DOMAIN is None
        assert girf.last_definitions_ is None


#**************************************************************************************************#
#                                  Class TestGradientNonlinearity                                   #
#**************************************************************************************************#
class TestGradientNonlinearity:
    """Requesting it must raise, never silently do nothing."""

    def test_raises_not_implemented_at_construction(self, tiny_seq_file):
        with pytest.raises(NotImplementedError, match="vendor"):
            GIRFArtifacts(seq_file=tiny_seq_file, include_gradient_nonlinearity=True)


#**************************************************************************************************#
#                                   Class TestConstructorValidation                                 #
#**************************************************************************************************#
class TestConstructorValidation:
    """Bad configuration is rejected up front."""

    def test_rejects_unknown_girf_mode(self, tiny_seq_file):
        with pytest.raises(ValueError, match="girf_mode"):
            GIRFArtifacts(seq_file=tiny_seq_file, girf_mode="not_a_mode")

    def test_measured_needs_exactly_one_source(self, tiny_seq_file):
        with pytest.raises(ValueError, match="measured"):
            GIRFArtifacts(seq_file=tiny_seq_file, girf_mode="measured")
        with pytest.raises(ValueError, match="measured"):
            GIRFArtifacts(seq_file=tiny_seq_file, girf_mode="measured",
                          bacon_data_dir="/nonexistent", measured_girf=object())


#**************************************************************************************************#
#                                   Class TestTrajectoryErrorOnly                                   #
#**************************************************************************************************#
class TestTrajectoryErrorOnly:
    """The trajectory-error-only path against a hand-rolled dual-pair NUFFT."""

    def test_matches_a_hand_rolled_dual_trajectory_nufft(self, tiny_seq_file):
        import girf_module as gmod
        from augmentrum.sampling.kspace_reconstructor import GriddingNUFFT

        out, girf = _run(tiny_seq_file, include_trajectory_error=True,
                         include_girf_phase=False, include_concomitant=False)

        mod = gmod.GIRFModule.from_synthetic(tiny_seq_file, seed=0)
        # GIRF-Sim always reports all 3 physical gradient/k-space axes, even
        # for an in-plane (single-slice) trajectory - the z column is just
        # ~0 throughout, not omitted - so this reference uses a 3-D NUFFT
        # with nz=1, exactly matching what GIRFArtifacts itself does.
        ndim = int(mod.k_nominal.shape[1])
        assert ndim == 3

        pts_nominal = mod.k_nominal.permute(0, 2, 1).reshape(-1, ndim).numpy()
        pts_actual = mod.k_actual.permute(0, 2, 1).reshape(-1, ndim).numpy()

        nx, ny, nz = MATRIX
        # Matches GIRFArtifacts._kmax_from_geometry with pixdim=PIXDIM_MM
        # (20 mm/voxel on every axis, including Z, even though nz=1) - not
        # the .seq file's own declared FOV, which _run deliberately
        # overrides (see PIXDIM_MM's comment).
        vox_m = PIXDIM_MM[0] / 1000.0
        kmax = np.array([(nx / 2.0) / (nx * vox_m), (ny / 2.0) / (ny * vox_m),
                         (nz / 2.0) / (nz * vox_m)])
        coords_nominal = (pts_nominal / (2.0 * kmax[None, :])).astype(np.float32)
        coords_actual = (pts_actual / (2.0 * kmax[None, :])).astype(np.float32)

        vol = _phantom()[0]                     # (nx, ny, nz, n_t)
        plane = vol.transpose(3, 0, 1, 2)        # (n_t, nx, ny, nz)

        nufft = GriddingNUFFT((nx, ny, nz), osf=2.0)
        kdata = nufft.forward(plane.astype(np.complex64), coords_actual)
        recon = nufft.adjoint(kdata, coords_nominal)   # (n_t, nx, ny, nz)
        expected = recon.transpose(1, 2, 3, 0)         # (nx, ny, nz, n_t)

        # GIRFArtifacts rescales its output to match the input phantom's
        # overall energy (_match_scale, same rationale as
        # FieldInhomogeneity/KspaceUndersampling); this hand-rolled
        # reference is a raw, unscaled adjoint, so compare shape only.
        out_unit = out[0] / np.linalg.norm(out[0])
        expected_unit = expected / np.linalg.norm(expected)
        num = np.linalg.norm(out_unit - expected_unit)
        assert num < 1e-4, f"relative error {num:.6f} too high"

    def test_off_differs_from_on(self, tiny_seq_file):
        out_off, _ = _run(tiny_seq_file, include_trajectory_error=False,
                          include_girf_phase=False, include_concomitant=False)
        out_on, _ = _run(tiny_seq_file, include_trajectory_error=True,
                         include_girf_phase=False, include_concomitant=False)
        assert not np.allclose(out_off, out_on)


#**************************************************************************************************#
#                                   Class TestSeedReproducibility                                   #
#**************************************************************************************************#
class TestSeedReproducibility:
    """A fixed girf_seed is deterministic; None draws a fresh realization."""

    def test_fixed_seed_is_bit_identical_across_calls(self, tiny_seq_file):
        out1, _ = _run(tiny_seq_file, girf_seed=7)
        out2, _ = _run(tiny_seq_file, girf_seed=7)
        np.testing.assert_array_equal(out1, out2)

    def test_no_seed_varies_across_calls(self, tiny_seq_file):
        out1, _ = _run(tiny_seq_file, girf_seed=None)
        out2, _ = _run(tiny_seq_file, girf_seed=None)
        assert not np.array_equal(out1, out2)


#**************************************************************************************************#
#                                 Class TestConcomitantAndSegments                                  #
#**************************************************************************************************#
class TestConcomitantAndSegments:
    """Concomitant-only ablation and severity-segment provenance/convergence."""

    def test_concomitant_only_changes_data_without_trajectory_error(self, tiny_seq_file):
        out_none, _ = _run(tiny_seq_file, include_trajectory_error=False,
                           include_girf_phase=False, include_concomitant=False)
        out_conc, girf = _run(tiny_seq_file, include_trajectory_error=False,
                              include_girf_phase=False, include_concomitant=True,
                              n_severity_segments=4)
        assert not np.allclose(out_none, out_conc)
        assert girf.last_severity_segments_ is not None
        assert girf.last_severity_segments_[0]['n_bins'] >= 1

    def test_more_segments_converge_toward_fine_reference(self, tiny_seq_file):
        reference, _ = _run(tiny_seq_file, include_trajectory_error=False,
                            include_girf_phase=False, include_concomitant=True,
                            n_severity_segments=64)

        errors = []
        for q in (1, 2, 8):
            out, _ = _run(tiny_seq_file, include_trajectory_error=False,
                          include_girf_phase=False, include_concomitant=True,
                          n_severity_segments=q)
            errors.append(float(np.linalg.norm(out - reference)))

        assert errors[-1] <= errors[0], f"error did not shrink with more segments: {errors}"

    def test_concomitant_needs_b0(self, tiny_seq_file):
        girf = GIRFArtifacts(seq_file=tiny_seq_file, b0_tesla=None,
                             include_trajectory_error=False, include_girf_phase=False,
                             include_concomitant=True)
        girf.b0_tesla = None   # force past the .seq file's own B0 definition
        with pytest.raises(ValueError, match="b0_tesla"):
            girf._b0_tesla(type('M', (), {'definitions': {}})())


#**************************************************************************************************#
#                                       Class TestMeasuredTier                                      #
#**************************************************************************************************#
class TestMeasuredTier:
    """A user-supplied measured GIRF (BaconGIRF.from_arrays), no Zenodo download."""

    def _tiny_bacon(self):
        import torch
        import bacon_girf as bg

        n_freq = 33
        H = (torch.randn(4, 3, n_freq) + 1j * torch.randn(4, 3, n_freq)) * 0.01
        freq_hz = torch.linspace(-500.0, 500.0, n_freq)
        return bg.BaconGIRF.from_arrays(H, freq_hz, gammabar_hz_per_mT=42576.0,
                                        adc_dwell_s=4e-5, order=1)

    def test_runs_end_to_end(self, tiny_seq_file):
        bacon = self._tiny_bacon()
        out, girf = _run(tiny_seq_file, girf_mode='measured', measured_girf=bacon,
                         include_trajectory_error=True, include_girf_phase=True,
                         include_concomitant=True, n_severity_segments=4)
        assert out.shape == _phantom().shape
        assert np.all(np.isfinite(out))

    def test_girf_phase_flag_has_no_effect_on_measured_tier(self, tiny_seq_file):
        bacon = self._tiny_bacon()
        out_with, _ = _run(tiny_seq_file, girf_mode='measured', measured_girf=bacon,
                           include_trajectory_error=True, include_girf_phase=True,
                           include_concomitant=False, girf_seed=None)
        out_without, _ = _run(tiny_seq_file, girf_mode='measured', measured_girf=bacon,
                              include_trajectory_error=True, include_girf_phase=False,
                              include_concomitant=False, girf_seed=None)
        np.testing.assert_array_equal(out_with, out_without)
