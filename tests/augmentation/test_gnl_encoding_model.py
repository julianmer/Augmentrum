####################################################################################################
#                                test_gnl_encoding_model.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-12                                                                              #
#                                                                                                  #
# Purpose: GNLEncodingModel against the 10 validation requirements in                              #
#          augmentrum_gnl_virtual_scanner_handover.md, using a tiny synthetic .seq fixture (same    #
#          pattern as test_girf_artifacts.py). Needs gradunwarp + GIRF-Sim; skipped otherwise.       #
#                                                                                                  #
####################################################################################################

"""
Tests for GNLEncodingModel (physically-direct gradient-nonlinearity encoding).
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
    import gradunwarp              # noqa: F401

    try:
        import mrs_seq              # noqa: F401
    except ImportError:
        repo_root = str(Path(girf_module.__file__).resolve().parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import mrs_seq              # noqa: F401

    from mrs_seq.system import build_system
    from mrs_seq.localization.steam import build_steam
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DEPS_AVAILABLE, reason="gradunwarp/GIRF-Sim not installed")

if DEPS_AVAILABLE:
    from augmentrum.augmentation.gnl_encoding_model import GNLEncodingModel
    from augmentrum.physics.gradient_nonlinearity import load_coefficients
    from augmentrum.physics.gnl_field import sample_coefficients

MATRIX = (4, 4, 1)
PIXDIM_MM = (20.0, 20.0, 20.0)


#**************#
#   fixtures   #
#**************#
@pytest.fixture(scope='module')
def tiny_seq_file(tmp_path_factory):
    """A slew-safe, deliberately tiny 3-shot spiral MRSI2D ".seq" file."""
    system = build_system('generic_3T', grad_raster_time=4e-5)
    seq, _ = build_steam(
        system, acquisition_mode='mrsi2d', voi_mm=(60.0, 60.0, 20.0),
        TE=0.020, TM=0.020, TR=0.2, fov_mm=(80.0, 80.0), matrix=(4, 4),
        n_shots=3, samples_per_shot=32, spectral_bandwidth_hz=2000.0, spectral_points=6,
        target_fid_duration=0.1, seq_name='tiny_gnl_test',
    )
    path = tmp_path_factory.mktemp('seq') / 'tiny_gnl_test.seq'
    seq.write(str(path))
    return str(path)


def _phantom(batch=1):
    nx, ny, nz = MATRIX
    n_t = 4
    yy, xx = np.mgrid[0:nx, 0:ny]
    disc = ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2 < (nx / 2.5) ** 2).astype(np.float32)
    t = np.arange(n_t) / 2000.0
    fid = np.exp(-8.0 * t).astype(np.complex64)
    vol = np.zeros((batch, nx, ny, nz, n_t), np.complex64)
    vol[:] = disc[None, :, :, None, None] * fid[None, None, None, None, :]
    return vol


def _run(seq_file, **kwargs):
    kwargs.setdefault('pixdim', PIXDIM_MM)
    gnl = GNLEncodingModel(seq_file=seq_file, **kwargs)
    out, _ = gnl.process_tensor(_phantom(), sw_hz=2000.0)
    return np.asarray(out), gnl


#**************************************************************************************************#
#                                Class TestZeroGNLReproducesBaseline                                #
#**************************************************************************************************#
class TestZeroGNLReproducesBaseline:
    """Requirement 1: include_gnl=False (and everything else off) reproduces
    the pre-GNL forward model exactly (bitwise identity - the module never
    runs)."""

    def test_everything_disabled_is_identity(self, tiny_seq_file):
        vol = _phantom()
        out, gnl = _run(tiny_seq_file, include_gnl=False, include_girf_phase=False,
                        include_concomitant=False, include_trajectory_error=False)
        assert np.array_equal(out, vol)
        assert gnl.DOMAIN is None
        assert gnl.last_definitions_ is None


#**************************************************************************************************#
#                                Class TestZeroCoefficientsAreNominal                                #
#**************************************************************************************************#
class TestZeroCoefficientsAreNominal:
    """Requirement 2: an all-zero (order 0/1 permitted) coefficient set
    contributes nothing - the reconstruction must be identical regardless of
    how many (all-zero) harmonic terms are carried, i.e. "pure linear terms,
    all zero" reproduces plain nominal encoding through the same NUFFT
    pipeline (not compared against the identity-bypass baseline, which skips
    the NUFFT round trip entirely and so is not a fair comparison - the
    NUFFT's own limited-sample reconstruction blur is present either way)."""

    def test_all_zero_coeffs_is_order_independent(self, tiny_seq_file):
        from gradunwarp.core.coeffs import Coeffs

        def null_model(n):
            zeros = lambda: np.zeros((n, n))
            return Coeffs(alpha_x=zeros(), alpha_y=zeros(), alpha_z=zeros(),
                         beta_x=zeros(), beta_y=zeros(), beta_z=zeros(), R0_m=0.25)

        out_order1, _ = _run(tiny_seq_file, include_gnl=True, truth_model=null_model(2),
                             max_harmonic_order=1, include_girf_phase=False,
                             include_concomitant=False, include_trajectory_error=False)
        out_order9, _ = _run(tiny_seq_file, include_gnl=True, truth_model=null_model(10),
                             max_harmonic_order=9, include_girf_phase=False,
                             include_concomitant=False, include_trajectory_error=False)

        np.testing.assert_allclose(out_order1, out_order9, atol=1e-5, rtol=1e-5)

    def test_zero_gnl_phase_directly(self, tiny_seq_file):
        """The phase term itself, not just the round-tripped image, is
        exactly zero for all-zero coefficients."""
        from gradunwarp.core.coeffs import Coeffs
        from augmentrum.physics.gnl_field import GradientNonlinearityPhase
        import torch

        zeros = lambda: np.zeros((4, 4))
        null_model = Coeffs(alpha_x=zeros(), alpha_y=zeros(), alpha_z=zeros(),
                           beta_x=zeros(), beta_y=zeros(), beta_z=zeros(), R0_m=0.25)
        rng = np.random.default_rng(0)
        positions = torch.tensor(rng.normal(scale=0.05, size=(10, 3)), dtype=torch.float32)
        k_traj = rng.normal(scale=15.0, size=(3, 20))
        term = GradientNonlinearityPhase(null_model, k_traj, max_order=3)
        phase = term.phase(positions, torch.arange(20, dtype=torch.float32))
        np.testing.assert_array_equal(phase.numpy(), np.zeros((10, 20)))


#**************************************************************************************************#
#                                  Class TestSingleHarmonicWiring                                   #
#**************************************************************************************************#
class TestSingleHarmonicWiring:
    """Requirement 3 (end-to-end wiring check - the physics itself is
    verified independently in tests/physics/test_gnl_field.py): a single
    nonzero harmonic changes the data, and simulate()'s direct API agrees
    with manually driving GradientNonlinearityPhase."""

    def test_single_harmonic_changes_data_and_matches_manual_phase(self, tiny_seq_file):
        from gradunwarp.core.coeffs import Coeffs
        import girf_mrsi_extensions as gmx
        from augmentrum.physics.gnl_field import GradientNonlinearityPhase

        zeros = lambda: np.zeros((4, 4))
        alpha_z = zeros()
        alpha_z[3, 0] = 0.05
        coeffs = Coeffs(alpha_x=zeros(), alpha_y=zeros(), alpha_z=alpha_z,
                       beta_x=zeros(), beta_y=zeros(), beta_z=zeros(), R0_m=0.25)

        gnl = GNLEncodingModel(seq_file=tiny_seq_file, truth_model=coeffs,
                               max_harmonic_order=3, pixdim=PIXDIM_MM)
        rho = np.ones(MATRIX[0] * MATRIX[1], dtype=np.complex64)
        phantom = gmx.SpectralPhantom.single_component(
            __import__('torch').as_tensor(rho), t2star_s=1e12, freq_hz=0.0)
        result = gnl.simulate(phantom, matrix=MATRIX)

        mod = gnl._load_girf_module()
        ndim = int(mod.k_nominal.shape[1])
        k_nominal = mod.k_nominal.permute(0, 2, 1).reshape(-1, ndim).numpy().astype(np.float64).T
        positions_np = gnl._positions(MATRIX, None).numpy()

        import torch
        term = GradientNonlinearityPhase(coeffs, k_nominal, max_order=3)
        positions_t = torch.tensor(positions_np, dtype=torch.float32)
        t_axis = torch.arange(k_nominal.shape[1], dtype=torch.float32)
        phi_manual = term.phase(positions_t, t_axis)

        expected_d = (torch.ones(positions_np.shape[0], dtype=torch.complex64).reshape(-1, 1)
                     * torch.exp(1j * phi_manual.to(torch.complex64)))
        E = gmx.nudft_encode_matrix(torch.tensor(k_nominal, dtype=torch.float32), positions_t)
        expected = (E * expected_d.transpose(0, 1)).sum(dim=1)

        np.testing.assert_allclose(result.d.numpy(), expected.numpy(), atol=1e-3, rtol=1e-3)
        assert not np.allclose(result.d.numpy(), 0.0)


#**************************************************************************************************#
#                                   Class TestMeasuredCalibration                                   #
#**************************************************************************************************#
class TestMeasuredCalibration:
    """Requirement 4: measured source loads and uses the calibration
    exactly (already verified at the field-reconstruction level in
    tests/physics/test_gnl_field.py - here just the wiring)."""

    def test_measured_source_populates_truth_model_from_the_file(self, tiny_seq_file):
        from augmentrum.physics.gradient_nonlinearity import default_coeff_file

        path = default_coeff_file()   # gradunwarp's own bundled test fixture
        out, gnl = _run(tiny_seq_file, source='measured', realism_level=2, calibration_path=path)
        expected = load_coefficients(path)
        np.testing.assert_array_equal(gnl.last_truth_model_.alpha_z, expected.alpha_z)
        assert np.all(np.isfinite(out))


#**************************************************************************************************#
#                                Class TestStochasticReproducibility                                #
#**************************************************************************************************#
class TestStochasticReproducibility:
    """Requirements 5, 6: same seed -> identical coefficients and data;
    different seeds -> different but plausible."""

    def test_same_seed_gives_identical_coefficients_and_data(self, tiny_seq_file):
        out1, gnl1 = _run(tiny_seq_file, source='stochastic', stochastic_seed=11,
                          max_harmonic_order=5)
        out2, gnl2 = _run(tiny_seq_file, source='stochastic', stochastic_seed=11,
                          max_harmonic_order=5)
        np.testing.assert_array_equal(gnl1.last_truth_model_.alpha_z, gnl2.last_truth_model_.alpha_z)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self, tiny_seq_file):
        out1, gnl1 = _run(tiny_seq_file, source='stochastic', stochastic_seed=1,
                          max_harmonic_order=5)
        out2, gnl2 = _run(tiny_seq_file, source='stochastic', stochastic_seed=2,
                          max_harmonic_order=5)
        assert not np.array_equal(gnl1.last_truth_model_.alpha_z, gnl2.last_truth_model_.alpha_z)
        assert not np.array_equal(out1, out2)


#**************************************************************************************************#
#                                    Class TestOrder9Stability                                      #
#**************************************************************************************************#
class TestOrder9Stability:
    """Requirement 7: max_harmonic_order=9 evaluates finite and stable."""

    def test_order_9_end_to_end(self, tiny_seq_file):
        out, gnl = _run(tiny_seq_file, source='stochastic', stochastic_seed=0,
                        max_harmonic_order=9, include_linear_terms=True)
        assert np.all(np.isfinite(out))
        assert gnl.last_truth_model_.alpha_z.shape == (10, 10)


#**************************************************************************************************#
#                                    Class TestRealismLevel4                                        #
#**************************************************************************************************#
class TestRealismLevel4:
    """Requirement 8: correction_model != truth_model by default at level 4,
    nonzero residual; levels 1-3 keep correction == truth by default."""

    def test_level_4_has_nonzero_residual_by_default(self, tiny_seq_file):
        calib = load_coefficients()
        _, gnl = _run(tiny_seq_file, truth_model=calib, realism_level=4, max_harmonic_order=9)
        positions = np.random.default_rng(0).normal(scale=0.05, size=(20, 3))
        residual = gnl.residual_displacement_m(positions)
        assert np.abs(residual).max() > 0.0

    def test_level_2_has_zero_residual_by_default(self, tiny_seq_file):
        calib = load_coefficients()
        _, gnl = _run(tiny_seq_file, truth_model=calib, realism_level=2, max_harmonic_order=9)
        positions = np.random.default_rng(0).normal(scale=0.05, size=(20, 3))
        residual = gnl.residual_displacement_m(positions)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_explicit_correction_model_overrides_the_level_4_default(self, tiny_seq_file):
        calib = load_coefficients()
        _, gnl = _run(tiny_seq_file, truth_model=calib, correction_model=calib,
                      realism_level=4, max_harmonic_order=9)
        assert gnl.last_correction_model_ is calib


#**************************************************************************************************#
#                                  Class TestComposesWithB0                                         #
#**************************************************************************************************#
class TestComposesWithB0:
    """Requirement 9: chained with the existing FieldInhomogeneity B0 model,
    both effects are present, neither double-applied."""

    def test_b0_and_gnl_both_present(self, tiny_seq_file):
        from augmentrum.augmentation.field_inhomogeneity import FieldInhomogeneity

        vol = _phantom()
        b0_map = np.full(MATRIX, 15.0, dtype=np.float64)
        b0 = FieldInhomogeneity(b0_map=b0_map, b0_n_segments=4)
        vol_b0, _ = b0.process_tensor(vol.copy(), sw_hz=2000.0)
        vol_b0 = np.asarray(vol_b0)
        assert not np.allclose(vol_b0, vol), "B0 alone should already change the data"

        gnl = GNLEncodingModel(seq_file=tiny_seq_file, source='stochastic',
                               stochastic_seed=0, max_harmonic_order=3, pixdim=PIXDIM_MM)
        out_b0_then_gnl, _ = gnl.process_tensor(vol_b0.copy(), sw_hz=2000.0)
        out_b0_then_gnl = np.asarray(out_b0_then_gnl)
        out_gnl_only, _ = gnl.process_tensor(vol.copy(), sw_hz=2000.0)
        out_gnl_only = np.asarray(out_gnl_only)

        # Combined output differs from GNL alone (B0 contribution is present)
        # and from B0 alone (GNL contribution is present).
        assert not np.allclose(out_b0_then_gnl, out_gnl_only)
        assert not np.allclose(out_b0_then_gnl, vol_b0)


#**************************************************************************************************#
#                                Class TestComposesWithGIRFSim                                      #
#**************************************************************************************************#
class TestComposesWithGIRFSim:
    """Requirement 10: GIRF eddy-current phase, concomitant phase, and GNL
    are all present simultaneously, none removed or duplicated."""

    def test_all_three_terms_each_contribute(self, tiny_seq_file):
        vol = _phantom()

        def run(**flags):
            out, _ = _run(tiny_seq_file, source='stochastic', stochastic_seed=0,
                          max_harmonic_order=3, b0_tesla=3.0, **flags)
            return out

        gnl_only = run(include_gnl=True, include_girf_phase=False,
                       include_concomitant=False, include_trajectory_error=False)
        girf_only = run(include_gnl=False, include_girf_phase=True,
                        include_concomitant=False, include_trajectory_error=False)
        conc_only = run(include_gnl=False, include_girf_phase=False,
                        include_concomitant=True, include_trajectory_error=False)
        traj_only = run(include_gnl=False, include_girf_phase=False,
                        include_concomitant=False, include_trajectory_error=True)
        combined = run(include_gnl=True, include_girf_phase=True,
                       include_concomitant=True, include_trajectory_error=True)

        # Every individual term must actually do something on its own...
        for name, out in (('gnl', gnl_only), ('girf', girf_only),
                          ('concomitant', conc_only), ('trajectory', traj_only)):
            assert not np.allclose(out, vol), f"{name}-only made no difference"

        # ...and the combination must differ from every subset (nothing
        # silently dominates/cancels/duplicates to reproduce a single term).
        for name, out in (('gnl', gnl_only), ('girf', girf_only),
                          ('concomitant', conc_only), ('trajectory', traj_only)):
            assert not np.allclose(combined, out), f"combined matches {name}-only"
