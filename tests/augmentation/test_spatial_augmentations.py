"""
Tests for SpatialAugmentations module.

Data layout
-----------
Every tensor here follows the **NIfTI-MRS convention** — spatial axes first,
channel-like axis (spectral points, or coils for the CSM pipeline) LAST:

    dim=3 :  (batch, X, Y, Z, C)
    dim=2 :  (batch, X, Y, C)

That is exactly what "NIfTI_MRS_Plus.get_data()" produces by stacking
"(X, Y, Z, T)" NIfTI arrays along a new leading batch axis.

Tests cover:
- Creation / initialization (incl. pixdim, padding_mode, chunk_channels,
  allow_rot90, translation_prob)
- Sampling augmentation specs (incl. the 'rot_axis' provenance key)
- 2D and 3D affine application in the NIfTI layout
- Exact identity at prob=0
- Flip axis semantics against torch.flip
- Complex data handling
- chunk_channels result-invariance
- Deterministic replay of a stored aug_spec_list
- SVS passthrough
- Anisotropic voxel correction via pixdim
- Coil subsampling along the last axis
- NIfTI list processing / tensor processing (BaseModule interface)
- Pipeline integration
- Route-pipeline convenience method
- Parameter injection from Augmentrum (zoom_min / zoom_max / etc.)
"""

import warnings

import pytest
import numpy as np
import torch

from augmentrum.augmentation.spatial_augmentations import SpatialAugmentations
from nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from augmentrum.core.pipeline import AugmentationPipeline

# fsl_mrs is noisy on NIfTI construction / header handling.
warnings.filterwarnings('ignore')


#*************#
#   helpers   #
#*************#
def make_spec(**overrides):
    """A fully-populated, do-nothing augmentation spec.

    Passing an explicit "aug_spec_list" bypasses sampling entirely, so this is
    how a test pins down one single transform (a flip, a 90° rotation, ...)
    without fighting the RNG.
    """
    spec = {
        'do_translate': False, 'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
        'do_z_rot': False, 'z_angle_deg': 0.0,
        'do_rot90': False, 'k_rot90': 0,
        'do_random_csm_rot': False, 'random_rot_deg': 0.0,
        'rot_axis': (0.0, 0.0, 1.0),
        'do_zoom': False,
        'do_shear': False, 'shear_xy': (0.0, 0.0), 'shear_z': 0.0,
        'do_flip': False, 'flip_x': False, 'flip_y': False, 'flip_z': False,
        'do_anisotropic': False, 'zoom_xyz': (1.0, 1.0, 1.0),
        'do_coil_sub': False, 'coil_keep': None,
        'pipeline': 'data',
    }
    spec.update(overrides)
    return spec


def physical_sphere(nx, ny, nz, pixdim, radius_mm):
    """A blob that is a sphere in *millimeters*.

    On an anisotropic grid its voxel-space footprint is an ellipsoid with
    semi-axes "radius_mm / pixdim_i" — the in-plane extent ratio is therefore
    "pixdim_y / pixdim_x".
    """
    dx, dy, dz = pixdim
    gx = (np.arange(nx) - (nx - 1) / 2.0) * dx
    gy = (np.arange(ny) - (ny - 1) / 2.0) * dy
    gz = (np.arange(nz) - (nz - 1) / 2.0) * dz
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
    r2 = X ** 2 + Y ** 2 + Z ** 2
    return np.exp(-r2 / (2.0 * radius_mm ** 2)).astype(np.float32)


def voxel_extents(vol):
    """Intensity-weighted RMS extent (in voxels) along each of the 3 axes."""
    v = np.clip(np.asarray(vol, dtype=np.float64), 0.0, None)
    total = v.sum()
    out = []
    for ax in range(3):
        others = tuple(a for a in range(3) if a != ax)
        profile = v.sum(axis=others)
        coords = np.arange(profile.size)
        mean = (profile * coords).sum() / total
        out.append(float(np.sqrt((profile * (coords - mean) ** 2).sum() / total)))
    return out


#***************************************************#
#   fixtures — nifti objects, shaped (x, y, z, t)   #
#***************************************************#
@pytest.fixture
def spatial_nifti_list():
    """Three NIfTI-MRS objects shaped (X, Y, Z, T) = (8, 8, 4, 16), so that
    stacking into a batch gives (3, 8, 8, 4, 16) — the 5-D NIfTI layout
    SpatialAugmentations(dim=3) expects.
    """
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    nifti_list = []
    for _ in range(3):
        data = (np.random.randn(8, 8, 4, 16)
                + 1j * np.random.randn(8, 8, 4, 16)).astype(np.complex64)
        nifti_list.append(gen_nifti_mrs(data, 1 / 2000, 123.0))
    return nifti_list


@pytest.fixture
def spatial_nifti_water():
    """Single water-ref NIfTI with matching (X, Y, Z, T) shape."""
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    data = (np.random.randn(8, 8, 4, 16)
            + 1j * np.random.randn(8, 8, 4, 16)).astype(np.complex64)
    return gen_nifti_mrs(data, 1 / 2000, 123.0)


@pytest.fixture
def spatial_2d():
    """Default 2-D spatial augmentation module."""
    return SpatialAugmentations(dim=2, prob=1.0)


@pytest.fixture
def spatial_3d():
    """Default 3-D spatial augmentation module."""
    return SpatialAugmentations(dim=3, prob=1.0)


@pytest.fixture
def batch_2d_real():
    """Real-valued 2-D batch tensor (B, X, Y, C)."""
    return torch.randn(4, 32, 32, 1, dtype=torch.float32)


@pytest.fixture
def batch_2d_complex():
    """Complex-valued 2-D batch tensor (B, X, Y, C)."""
    return (torch.randn(4, 32, 32, 1) + 1j * torch.randn(4, 32, 32, 1)).to(torch.complex64)


@pytest.fixture
def batch_3d_real():
    """Real-valued 3-D batch tensor (B, X, Y, Z, C)."""
    return torch.randn(2, 16, 16, 8, 1, dtype=torch.float32)


@pytest.fixture
def batch_3d_complex():
    """Complex-valued 3-D batch tensor (B, X, Y, Z, C)."""
    return (torch.randn(2, 16, 16, 8, 1)
            + 1j * torch.randn(2, 16, 16, 8, 1)).to(torch.complex64)


@pytest.fixture
def csm_3d():
    """Multi-coil 3-D complex tensor for the CSM pipeline — coils LAST."""
    return (torch.randn(2, 16, 16, 8, 8)
            + 1j * torch.randn(2, 16, 16, 8, 8)).to(torch.complex64)


#**************************************************************************************************#
#                                    Class TestSpatialCreation                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test SpatialAugmentations initialization.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestSpatialCreation:
    """Test SpatialAugmentations initialization."""

    def test_defaults(self):
        aug = SpatialAugmentations()
        assert aug.dim == 3
        assert aug.prob == 0.5
        assert aug.pipeline == 'data'
        assert aug.zoom_min == 0.9
        assert aug.zoom_max == 1.1

    def test_new_geometry_defaults(self):
        """The resampling defaults are part of the public contract."""
        aug = SpatialAugmentations()
        assert aug.pixdim is None
        assert aug.padding_mode == 'zeros'   # was hardcoded 'border'
        assert aug.chunk_channels == 64
        assert aug.allow_rot90 is True
        assert aug.translation_prob is None

    def test_custom_params(self):
        aug = SpatialAugmentations(
            dim=2, prob=0.8, max_z_angle_deg=45.0,
            zoom_min=0.8, zoom_max=1.2, shear_max=0.3,
            min_coils=4, max_coils=10, pipeline='csm',
        )
        assert aug.dim == 2
        assert aug.prob == 0.8
        assert aug.max_z_angle_deg == 45.0
        assert aug.zoom_min == 0.8
        assert aug.zoom_max == 1.2
        assert aug.shear_max == 0.3
        assert aug.min_coils == 4
        assert aug.max_coils == 10
        assert aug.pipeline == 'csm'

    def test_custom_geometry_params(self):
        aug = SpatialAugmentations(dim=3, pixdim=(2.8, 3.5, 4.0),
                                   padding_mode='border', chunk_channels=None,
                                   allow_rot90=False, translation_prob=0.25)
        assert aug.pixdim == (2.8, 3.5, 4.0)
        assert aug.padding_mode == 'border'
        assert aug.chunk_channels is None
        assert aug.allow_rot90 is False
        assert aug.translation_prob == 0.25

    def test_pixdim_coerced_to_float_tuple(self):
        aug = SpatialAugmentations(dim=2, pixdim=[2, 3])
        assert aug.pixdim == (2.0, 3.0)
        assert all(isinstance(v, float) for v in aug.pixdim)

    def test_pixdim_too_short_raises(self):
        with pytest.raises(ValueError, match='pixdim'):
            SpatialAugmentations(dim=3, pixdim=(1.0, 2.0))

    def test_invalid_dim(self):
        with pytest.raises(AssertionError):
            SpatialAugmentations(dim=4)

    def test_data_ranges_property(self):
        aug = SpatialAugmentations(zoom_min=0.8, zoom_max=1.3, shear_max=0.2)
        r = aug.data_ranges
        assert r['zoom_range'] == (0.8, 1.3)
        assert r['shear_max'] == 0.2

    def test_ranges_expose_translation_prob(self):
        aug = SpatialAugmentations(translation_prob=0.3)
        assert aug.data_ranges['translation_prob'] == 0.3
        assert aug.csm_ranges['translation_prob'] == 0.3

    def test_csm_ranges_uses_csm_angle(self):
        aug = SpatialAugmentations(max_z_angle_deg=10.0, csm_max_z_angle_deg=180.0)
        assert aug.data_ranges['max_z_angle_deg'] == 10.0
        assert aug.csm_ranges['max_z_angle_deg'] == 180.0

    def test_csm_angle_mirrors_data_when_none(self):
        aug = SpatialAugmentations(max_z_angle_deg=25.0, csm_max_z_angle_deg=None)
        assert aug.csm_ranges['max_z_angle_deg'] == 25.0

    def test_bulk_override_dicts(self):
        aug = SpatialAugmentations(data_ranges={'shear_max': 0.99})
        assert aug.data_ranges['shear_max'] == 0.99  # override wins

    def test_supported_backends(self):
        aug = SpatialAugmentations()
        assert aug.supports_backend(Backend.NIFTI_LIST)
        assert aug.supports_backend(Backend.NUMPY)
        assert aug.supports_backend(Backend.PYTORCH)


#**************************************************************************************************#
#                                    Class TestSpatialSampling                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test augmentation spec sampling.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class TestSpatialSampling:
    """Test augmentation spec sampling."""

    def test_sample_returns_correct_count(self, spatial_3d):
        specs = spatial_3d.sample_augmentations(8)
        assert len(specs) == 8

    def test_sample_contains_required_keys(self, spatial_3d):
        spec = spatial_3d.sample_augmentations(1)[0]
        required = {
            'do_translate', 'tx', 'ty', 'tz',
            'do_z_rot', 'z_angle_deg',
            'do_rot90', 'k_rot90',
            'do_random_csm_rot', 'random_rot_deg', 'rot_axis',
            'do_zoom',
            'do_shear', 'shear_xy', 'shear_z',
            'do_flip', 'flip_x', 'flip_y', 'flip_z',
            'do_anisotropic', 'zoom_xyz',
            'do_coil_sub', 'coil_keep',
            'pipeline',
        }
        assert required.issubset(spec.keys())

    def test_rot_axis_is_unit_length_when_rotating(self):
        """The random rotation axis lives in the spec, so a saved spec replays."""
        aug = SpatialAugmentations(dim=3, prob=1.0)
        for spec in aug.sample_augmentations(20):
            assert spec['do_random_csm_rot']
            axis = np.asarray(spec['rot_axis'], dtype=float)
            assert axis.shape == (3,)
            assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-5)

    def test_rot_axis_default_when_not_rotating(self):
        aug = SpatialAugmentations(dim=3, prob=0.0)
        for spec in aug.sample_augmentations(5):
            assert spec['rot_axis'] == (0.0, 0.0, 1.0)

    def test_sample_pipeline_tag(self, spatial_3d):
        spec_d = spatial_3d.sample_augmentations(1, pipeline='data')[0]
        spec_c = spatial_3d.sample_augmentations(1, pipeline='csm')[0]
        assert spec_d['pipeline'] == 'data'
        assert spec_c['pipeline'] == 'csm'

    def test_prob_zero_disables_all(self):
        aug = SpatialAugmentations(prob=0.0)
        specs = aug.sample_augmentations(50)
        for s in specs:
            assert not s['do_translate']
            assert not s['do_z_rot']
            assert not s['do_rot90']
            assert not s['do_random_csm_rot']
            assert not s['do_zoom']
            assert not s['do_shear']
            assert not s['do_flip']
            assert not s['do_anisotropic']
            assert not s['do_coil_sub']

    def test_prob_one_enables_all(self):
        aug = SpatialAugmentations(prob=1.0)
        specs = aug.sample_augmentations(50)
        for s in specs:
            assert s['do_translate']
            assert s['do_z_rot']
            assert s['do_zoom']
            assert s['do_shear']
            assert s['do_flip']

    def test_zoom_within_range_on_every_axis(self):
        aug = SpatialAugmentations(prob=1.0, zoom_min=0.8, zoom_max=1.2)
        for s in aug.sample_augmentations(100):
            for v in s['zoom_xyz']:
                assert 0.8 <= v <= 1.2

    def test_zoom_does_not_compound_past_its_range(self):
        """
        Zoom and anisotropic scaling used to be separate multiplicative
        parameters, so zoom_max=1.1 with scale_max=1.1 reached 1.21. One per-axis
        range now bounds the whole transform.
        """
        aug = SpatialAugmentations(prob=1.0, zoom_min=0.9, zoom_max=1.1)
        peak = max(v for s in aug.sample_augmentations(300) for v in s['zoom_xyz'])
        assert peak <= 1.1 + 1e-9, f"reachable zoom {peak} exceeds zoom_max"

    def test_isotropic_zoom_is_the_special_case(self):
        """
        do_zoom alone draws once and broadcasts (ax = ay = az); do_anisotropic
        draws each axis. Sampled at prob=0.5 so both branches occur, then split
        on the flags rather than forced, which is what the sampler actually does.
        """
        aug = SpatialAugmentations(dim=3, prob=0.5, zoom_min=0.8, zoom_max=1.2)
        specs = aug.sample_augmentations(600)

        iso = [s for s in specs if s['do_zoom'] and not s['do_anisotropic']]
        aniso = [s for s in specs if s['do_anisotropic']]
        assert iso and aniso, "prob=0.5 should produce both branches"

        for s in iso:
            assert len(set(s['zoom_xyz'])) == 1, f"isotropic branch gave {s['zoom_xyz']}"
        assert any(len(set(s['zoom_xyz'])) > 1 for s in aniso), \
            "anisotropic branch never produced differing axes"

        off = [s for s in specs if not s['do_zoom'] and not s['do_anisotropic']]
        for s in off:
            assert s['zoom_xyz'] == (1.0, 1.0, 1.0)

    def test_allow_rot90_false_never_samples_rot90(self):
        aug = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=False)
        specs = aug.sample_augmentations(200)
        assert not any(s['do_rot90'] for s in specs)
        assert all(s['k_rot90'] == 0 for s in specs)

    def test_allow_rot90_true_samples_rot90(self):
        aug = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=True)
        specs = aug.sample_augmentations(200)
        assert all(s['do_rot90'] for s in specs)
        assert all(s['k_rot90'] in (1, 2, 3) for s in specs)


#**************************************************************************************************#
#                                    Class TestTranslationProb                                     #
#**************************************************************************************************#
#                                                                                                  #
# translation_prob decouples translation from the master `prob`.                                   #
#                                                                                                  #
#**************************************************************************************************#
class TestTranslationProb:
    """translation_prob decouples translation from the master `prob`."""

    N_DRAWS = 2000

    def _frac(self, aug):
        specs = aug.sample_augmentations(self.N_DRAWS)
        return float(np.mean([s['do_translate'] for s in specs])), specs

    def test_none_falls_back_to_prob(self):
        aug = SpatialAugmentations(dim=3, prob=0.5, translation_prob=None)
        frac, _ = self._frac(aug)
        assert 0.45 < frac < 0.55

    def test_honoured_when_prob_positive(self):
        aug = SpatialAugmentations(dim=3, prob=0.5, translation_prob=0.9)
        frac, specs = self._frac(aug)
        assert 0.86 < frac < 0.94
        # the other augmentations still follow `prob`
        z_frac = float(np.mean([s['do_z_rot'] for s in specs]))
        assert 0.45 < z_frac < 0.55

    def test_low_translation_prob(self):
        aug = SpatialAugmentations(dim=3, prob=1.0, translation_prob=0.1)
        frac, specs = self._frac(aug)
        assert 0.07 < frac < 0.13
        assert all(s['do_z_rot'] for s in specs)  # prob=1 elsewhere

    def test_prob_zero_overrides_translation_prob(self):
        """prob=0 is the master switch — a zero probability always means identity."""
        aug = SpatialAugmentations(dim=3, prob=0.0, translation_prob=1.0)
        frac, _ = self._frac(aug)
        assert frac == 0.0


#**************************************************************************************************#
#                                        Class TestIdentity                                        #
#**************************************************************************************************#
#                                                                                                  #
# prob=0 must be a bit-exact pass-through, every single time.                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestIdentity:
    """prob=0 must be a bit-exact pass-through, every single time.

    A previous bug let translation slip through intermittently, so this is
    checked with torch.equal (not allclose) and repeated over many trials.
    """

    N_TRIALS = 25

    def test_identity_5d_exact(self):
        aug = SpatialAugmentations(dim=3, prob=0.0)
        for _ in range(self.N_TRIALS):
            x = torch.randn(2, 8, 6, 4, 5, dtype=torch.float32)
            out, _ = aug.apply(x)
            assert torch.equal(out, x)

    def test_identity_4d_exact(self):
        aug = SpatialAugmentations(dim=2, prob=0.0)
        for _ in range(self.N_TRIALS):
            x = torch.randn(3, 12, 10, 5, dtype=torch.float32)
            out, _ = aug.apply(x)
            assert torch.equal(out, x)

    def test_identity_complex_exact(self):
        aug = SpatialAugmentations(dim=3, prob=0.0)
        for _ in range(self.N_TRIALS):
            x = torch.randn(2, 8, 6, 4, 5, dtype=torch.complex64)
            out, _ = aug.apply(x)
            assert torch.equal(out, x)

    def test_identity_exact_with_pixdim(self):
        """The anisotropy correction must not perturb the identity affine."""
        aug = SpatialAugmentations(dim=3, prob=0.0, pixdim=(2.8, 3.5, 4.0))
        for _ in range(self.N_TRIALS):
            x = torch.randn(2, 8, 6, 4, 5, dtype=torch.float32)
            out, _ = aug.apply(x)
            assert torch.equal(out, x)

    def test_identity_exact_with_translation_prob(self):
        aug = SpatialAugmentations(dim=3, prob=0.0, translation_prob=1.0)
        for _ in range(self.N_TRIALS):
            x = torch.randn(2, 8, 6, 4, 5, dtype=torch.float32)
            out, _ = aug.apply(x)
            assert torch.equal(out, x)

    def test_identity_fresh_instances(self):
        """Also exact for a brand-new module each trial (no cached state)."""
        for _ in range(self.N_TRIALS):
            aug = SpatialAugmentations(dim=3, prob=0.0)
            x = torch.randn(1, 8, 8, 4, 3, dtype=torch.float32)
            out, _ = aug.apply(x)
            assert torch.equal(out, x)


#**************************************************************************************************#
#                                        Class TestFlipAxes                                        #
#**************************************************************************************************#
#                                                                                                  #
# flip_x / flip_y / flip_z act on NIfTI axes 1 / 2 / 3.                                            #
#                                                                                                  #
#**************************************************************************************************#
class TestFlipAxes:
    """flip_x / flip_y / flip_z act on NIfTI axes 1 / 2 / 3."""

    ATOL = 1e-5

    @pytest.mark.parametrize('flag,axis', [('flip_x', 1), ('flip_y', 2), ('flip_z', 3)])
    def test_flip_matches_torch_flip_3d(self, flag, axis):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        x = torch.randn(2, 8, 6, 4, 3, dtype=torch.float32)
        spec = make_spec(do_flip=True, **{flag: True})
        out, _ = aug.apply(x, aug_spec_list=[spec, spec])
        assert torch.allclose(out, torch.flip(x, dims=[axis]), atol=self.ATOL)

    @pytest.mark.parametrize('flag,axis', [('flip_x', 1), ('flip_y', 2)])
    def test_flip_matches_torch_flip_2d(self, flag, axis):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        x = torch.randn(2, 8, 6, 3, dtype=torch.float32)
        spec = make_spec(do_flip=True, **{flag: True})
        out, _ = aug.apply(x, aug_spec_list=[spec, spec])
        assert torch.allclose(out, torch.flip(x, dims=[axis]), atol=self.ATOL)

    def test_flip_axes_are_distinct(self):
        """A flip on one axis must not coincide with a flip on another."""
        aug = SpatialAugmentations(dim=3, prob=1.0)
        x = torch.randn(1, 8, 6, 4, 2, dtype=torch.float32)
        outs = {}
        for flag in ('flip_x', 'flip_y', 'flip_z'):
            spec = make_spec(do_flip=True, **{flag: True})
            outs[flag], _ = aug.apply(x, aug_spec_list=[spec])
        assert not torch.allclose(outs['flip_x'], outs['flip_y'], atol=self.ATOL)
        assert not torch.allclose(outs['flip_x'], outs['flip_z'], atol=self.ATOL)
        assert not torch.allclose(outs['flip_y'], outs['flip_z'], atol=self.ATOL)

    def test_all_flips_together(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        x = torch.randn(1, 8, 6, 4, 2, dtype=torch.float32)
        spec = make_spec(do_flip=True, flip_x=True, flip_y=True, flip_z=True)
        out, _ = aug.apply(x, aug_spec_list=[spec])
        assert torch.allclose(out, torch.flip(x, dims=[1, 2, 3]), atol=self.ATOL)


#**************************************************************************************************#
#                                     Class TestSpatialApply2D                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test apply() on 2-D data — (B, X, Y, C).                                                         #
#                                                                                                  #
#**************************************************************************************************#
class TestSpatialApply2D:
    """Test apply() on 2-D data — (B, X, Y, C)."""

    def test_output_shape(self, spatial_2d, batch_2d_real):
        out, specs = spatial_2d.apply(batch_2d_real)
        assert out.shape == batch_2d_real.shape
        assert len(specs) == batch_2d_real.shape[0]

    def test_complex_output(self, spatial_2d, batch_2d_complex):
        out, _ = spatial_2d.apply(batch_2d_complex)
        assert out.shape == batch_2d_complex.shape
        assert out.is_complex()

    def test_data_changes(self, spatial_2d, batch_2d_real):
        out, _ = spatial_2d.apply(batch_2d_real)
        assert not torch.allclose(out, batch_2d_real)

    def test_pre_specified_specs(self, spatial_2d, batch_2d_real):
        specs = spatial_2d.sample_augmentations(batch_2d_real.shape[0])
        out, returned_specs = spatial_2d.apply(batch_2d_real, aug_spec_list=specs)
        assert returned_specs is specs
        assert out.shape == batch_2d_real.shape

    def test_identity_when_all_off(self, batch_2d_real):
        aug = SpatialAugmentations(dim=2, prob=0.0)
        out, _ = aug.apply(batch_2d_real)
        assert torch.equal(out, batch_2d_real)

    def test_many_channels(self):
        """The spectral axis rides along as channels."""
        aug = SpatialAugmentations(dim=2, prob=1.0)
        x = torch.randn(1, 16, 16, 128, dtype=torch.complex64)
        out, _ = aug.apply(x)
        assert out.shape == x.shape


#**************************************************************************************************#
#                                     Class TestSpatialApply3D                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test apply() on 3-D data — (B, X, Y, Z, C).                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestSpatialApply3D:
    """Test apply() on 3-D data — (B, X, Y, Z, C)."""

    def test_output_shape(self, spatial_3d, batch_3d_real):
        out, specs = spatial_3d.apply(batch_3d_real)
        assert out.shape == batch_3d_real.shape
        assert len(specs) == batch_3d_real.shape[0]

    def test_complex_output(self, spatial_3d, batch_3d_complex):
        out, _ = spatial_3d.apply(batch_3d_complex)
        assert out.shape == batch_3d_complex.shape
        assert out.is_complex()

    def test_data_changes(self, spatial_3d, batch_3d_real):
        out, _ = spatial_3d.apply(batch_3d_real)
        assert not torch.allclose(out, batch_3d_real)

    def test_identity_when_all_off(self, batch_3d_real):
        aug = SpatialAugmentations(dim=3, prob=0.0)
        out, _ = aug.apply(batch_3d_real)
        assert torch.equal(out, batch_3d_real)

    def test_mrsi_volume_spectral_axis_last(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        mrsi = torch.randn(1, 16, 16, 4, 256, dtype=torch.complex64)
        out, _ = aug.apply(mrsi)
        assert out.shape == mrsi.shape

    def test_all_channels_share_one_transform(self):
        """The spatial transform must not vary with spectral point."""
        aug = SpatialAugmentations(dim=3, prob=1.0)
        base = torch.randn(1, 8, 8, 4, 1, dtype=torch.float32)
        # channel c is the same volume scaled by (c + 1)
        scales = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 1, 1, 4)
        x = base * scales
        out, _ = aug.apply(x)
        for c in range(1, 4):
            assert torch.allclose(out[..., c], out[..., 0] * (c + 1), atol=1e-4)


#**************************************************************************************************#
#                                      Class TestPaddingMode                                       #
#**************************************************************************************************#
#                                                                                                  #
# padding_mode is now configurable and defaults to 'zeros'.                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestPaddingMode:
    """padding_mode is now configurable and defaults to 'zeros'."""

    TRANSLATE = dict(do_translate=True, tx=0.9)

    def test_zeros_pads_with_zeros(self):
        aug = SpatialAugmentations(dim=3, prob=1.0, padding_mode='zeros')
        x = torch.ones(1, 16, 16, 4, 2, dtype=torch.float32)
        out, _ = aug.apply(x, aug_spec_list=[make_spec(**self.TRANSLATE)])
        assert float((out == 0).float().mean()) > 0.1

    def test_border_extrudes_edges(self):
        aug = SpatialAugmentations(dim=3, prob=1.0, padding_mode='border')
        x = torch.ones(1, 16, 16, 4, 2, dtype=torch.float32)
        out, _ = aug.apply(x, aug_spec_list=[make_spec(**self.TRANSLATE)])
        assert torch.allclose(out, x, atol=1e-5)


#**************************************************************************************************#
#                                    Class TestComplexHandling                                     #
#**************************************************************************************************#
#                                                                                                  #
# Complex input is resampled component-wise.                                                       #
#                                                                                                  #
#**************************************************************************************************#
class TestComplexHandling:
    """Complex input is resampled component-wise."""

    def test_complex_equals_real_path(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        specs = aug.sample_augmentations(2)
        real = torch.randn(2, 12, 12, 4, 3, dtype=torch.float32)
        imag = torch.randn(2, 12, 12, 4, 3, dtype=torch.float32)
        x = torch.complex(real, imag)

        out_c, _ = aug.apply(x, aug_spec_list=specs)
        out_r, _ = aug.apply(real, aug_spec_list=specs)
        out_i, _ = aug.apply(imag, aug_spec_list=specs)

        assert out_c.is_complex()
        assert torch.allclose(out_c.real, out_r, atol=1e-6)
        assert torch.allclose(out_c.imag, out_i, atol=1e-6)

    def test_complex_2d_equals_real_path(self):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        specs = aug.sample_augmentations(2)
        real = torch.randn(2, 16, 16, 3, dtype=torch.float32)
        x = torch.complex(real, torch.zeros_like(real))
        out_c, _ = aug.apply(x, aug_spec_list=specs)
        out_r, _ = aug.apply(real, aug_spec_list=specs)
        assert torch.allclose(out_c.real, out_r, atol=1e-6)
        assert torch.allclose(out_c.imag, torch.zeros_like(real), atol=1e-6)

    def test_complex128_input(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        x = torch.randn(1, 8, 8, 4, 2, dtype=torch.complex128)
        out, _ = aug.apply(x)
        assert out.shape == x.shape
        assert out.is_complex()


#**************************************************************************************************#
#                                     Class TestChunkChannels                                      #
#**************************************************************************************************#
#                                                                                                  #
# chunk_channels bounds peak memory; it must never change the result.                              #
#                                                                                                  #
#**************************************************************************************************#
class TestChunkChannels:
    """chunk_channels bounds peak memory; it must never change the result."""

    CHUNKS = [None, 1, 2, 64]

    def _reference(self, x, specs, dim=3):
        aug = SpatialAugmentations(dim=dim, prob=1.0, chunk_channels=None)
        out, _ = aug.apply(x, aug_spec_list=specs)
        return out

    def test_real_invariance(self):
        specs = SpatialAugmentations(dim=3, prob=1.0).sample_augmentations(2)
        x = torch.randn(2, 12, 12, 4, 5, dtype=torch.float32)
        ref = self._reference(x, specs)
        for chunk in self.CHUNKS:
            aug = SpatialAugmentations(dim=3, prob=1.0, chunk_channels=chunk)
            out, _ = aug.apply(x, aug_spec_list=specs)
            assert torch.equal(out, ref), f'chunk_channels={chunk} changed the result'

    def test_complex_invariance(self):
        specs = SpatialAugmentations(dim=3, prob=1.0).sample_augmentations(2)
        x = (torch.randn(2, 12, 12, 4, 5)
             + 1j * torch.randn(2, 12, 12, 4, 5)).to(torch.complex64)
        ref = self._reference(x, specs)
        for chunk in self.CHUNKS:
            aug = SpatialAugmentations(dim=3, prob=1.0, chunk_channels=chunk)
            out, _ = aug.apply(x, aug_spec_list=specs)
            assert torch.equal(out, ref), f'chunk_channels={chunk} changed the result'

    def test_chunk_larger_than_channel_count(self):
        specs = SpatialAugmentations(dim=3, prob=1.0).sample_augmentations(1)
        x = torch.randn(1, 8, 8, 4, 3, dtype=torch.float32)
        ref = self._reference(x, specs)
        aug = SpatialAugmentations(dim=3, prob=1.0, chunk_channels=1024)
        out, _ = aug.apply(x, aug_spec_list=specs)
        assert torch.equal(out, ref)

    def test_2d_invariance(self):
        specs = SpatialAugmentations(dim=2, prob=1.0).sample_augmentations(2)
        x = torch.randn(2, 16, 16, 5, dtype=torch.float32)
        ref = self._reference(x, specs, dim=2)
        for chunk in self.CHUNKS:
            aug = SpatialAugmentations(dim=2, prob=1.0, chunk_channels=chunk)
            out, _ = aug.apply(x, aug_spec_list=specs)
            assert torch.equal(out, ref), f'chunk_channels={chunk} changed the result'


#**************************************************************************************************#
#                                       Class TestSpecReplay                                       #
#**************************************************************************************************#
#                                                                                                  #
# A saved aug_spec_list must reproduce the transform bit-for-bit.                                  #
#                                                                                                  #
#**************************************************************************************************#
class TestSpecReplay:
    """A saved aug_spec_list must reproduce the transform bit-for-bit.

    Data and its label maps are augmented in separate calls; if any parameter
    were re-sampled inside apply() they would drift apart. 'rot_axis' is the
    key that used to be missing.
    """

    def test_replay_is_bit_identical_3d(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        specs = aug.sample_augmentations(2, pipeline='csm')
        x = torch.randn(2, 12, 12, 4, 3, dtype=torch.complex64)
        out_a, _ = aug.apply(x, aug_spec_list=specs)
        out_b, _ = aug.apply(x, aug_spec_list=specs)
        assert torch.equal(out_a, out_b)

    def test_replay_is_bit_identical_2d(self):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        specs = aug.sample_augmentations(2)
        x = torch.randn(2, 16, 16, 3, dtype=torch.float32)
        out_a, _ = aug.apply(x, aug_spec_list=specs)
        out_b, _ = aug.apply(x, aug_spec_list=specs)
        assert torch.equal(out_a, out_b)

    def test_replay_across_instances(self):
        """A spec dict is portable — a second module reproduces it exactly."""
        sampler = SpatialAugmentations(dim=3, prob=1.0)
        specs = sampler.sample_augmentations(2, pipeline='csm')
        x = torch.randn(2, 12, 12, 4, 3, dtype=torch.float32)
        out_a, _ = sampler.apply(x, aug_spec_list=specs)
        replayer = SpatialAugmentations(dim=3, prob=0.0)
        out_b, _ = replayer.apply(x, aug_spec_list=specs)
        assert torch.equal(out_a, out_b)

    def test_random_rotation_uses_spec_axis(self):
        """Changing only rot_axis changes the output — proof it is really used."""
        aug = SpatialAugmentations(dim=3, prob=1.0)
        x = torch.randn(1, 12, 12, 8, 2, dtype=torch.float32)
        s1 = make_spec(do_random_csm_rot=True, random_rot_deg=45.0,
                       rot_axis=(0.0, 0.0, 1.0))
        s2 = make_spec(do_random_csm_rot=True, random_rot_deg=45.0,
                       rot_axis=(1.0, 0.0, 0.0))
        out1, _ = aug.apply(x, aug_spec_list=[s1])
        out2, _ = aug.apply(x, aug_spec_list=[s2])
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_replayed_data_and_label_agree(self):
        """The canonical use-case: one spec list, two tensors, same geometry."""
        aug = SpatialAugmentations(dim=3, prob=1.0)
        specs = aug.sample_augmentations(1, pipeline='csm')
        vol = torch.zeros(1, 16, 16, 8, 1, dtype=torch.float32)
        vol[0, 4:9, 5:11, 2:6, 0] = 1.0
        label = vol.clone()
        out_vol, _ = aug.apply(vol, aug_spec_list=specs)
        out_lab, _ = aug.apply(label, aug_spec_list=specs)
        assert torch.equal(out_vol, out_lab)


#**************************************************************************************************#
#                                     Class TestSVSPassthrough                                     #
#**************************************************************************************************#
#                                                                                                  #
# Single-voxel spectroscopy has no spatial extent to resample.                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestSVSPassthrough:
    """Single-voxel spectroscopy has no spatial extent to resample."""

    def test_svs_untouched(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        svs = torch.randn(3, 1, 1, 1, 2048, dtype=torch.complex64)
        out, specs = aug.apply(svs)
        assert torch.equal(out, svs)
        assert len(specs) == 3

    def test_svs_untouched_2d(self):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        svs = torch.randn(2, 1, 1, 512, dtype=torch.complex64)
        out, _ = aug.apply(svs)
        assert torch.equal(out, svs)

    def test_svs_stores_specs(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        svs = torch.randn(2, 1, 1, 1, 64, dtype=torch.complex64)
        aug.apply(svs)
        assert aug.aug_specs_ is not None
        assert len(aug.aug_specs_) == 2

    def test_single_slice_volume_is_still_resampled(self):
        """Only *fully* singleton spatial axes trigger the passthrough."""
        aug = SpatialAugmentations(dim=3, prob=1.0)
        x = torch.randn(1, 16, 16, 1, 4, dtype=torch.float32)
        out, _ = aug.apply(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)


#**************************************************************************************************#
#                                       Class TestAnisotropy                                       #
#**************************************************************************************************#
#                                                                                                  #
# pixdim makes rotations physical on an anisotropic grid.                                          #
#                                                                                                  #
#**************************************************************************************************#
class TestAnisotropy:
    """pixdim makes rotations physical on an anisotropic grid.

    Setup: a square 48 x 48 voxel matrix with pixdim (1.0, 2.0, 1.0) mm, i.e. a
    48 x 96 mm field of view, holding a blob that is a sphere in millimeters.
    In voxel space that blob is an ellipsoid whose in-plane extent ratio equals
    pixdim_y / pixdim_x = 2.

    An explicit 90 degree z-rotation is used (allow_rot90=False so nothing else
    is sampled and no non-square-FOV rot90 warning fires).
    """

    NX = NY = 48
    NZ = 8
    PIXDIM = (1.0, 2.0, 1.0)
    RADIUS_MM = 6.0

    @pytest.fixture
    def sphere_volume(self):
        blob = physical_sphere(self.NX, self.NY, self.NZ, self.PIXDIM, self.RADIUS_MM)
        return torch.from_numpy(blob)[None, ..., None]  # (1, X, Y, Z, 1)

    @staticmethod
    def _rot90_spec():
        return make_spec(do_z_rot=True, z_angle_deg=90.0)

    def _extents(self, out):
        return voxel_extents(out[0, ..., 0].detach().numpy())

    def test_input_is_physically_spherical(self, sphere_volume):
        ex, ey, _ = self._extents(sphere_volume)
        # voxel extents differ by pixdim ratio ...
        assert np.isclose(ex / ey, self.PIXDIM[1] / self.PIXDIM[0], rtol=0.05)
        # ... which is exactly what makes the physical extents equal
        assert np.isclose(ex * self.PIXDIM[0], ey * self.PIXDIM[1], rtol=0.05)

    def test_with_pixdim_sphere_stays_spherical(self, sphere_volume):
        aug = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=False,
                                   pixdim=self.PIXDIM)
        out, _ = aug.apply(sphere_volume, aug_spec_list=[self._rot90_spec()])
        ex, ey, _ = self._extents(out)
        # physical extents still match => still a sphere in mm
        assert np.isclose(ex * self.PIXDIM[0], ey * self.PIXDIM[1], rtol=0.05)
        # equivalently, the voxel-space ratio is preserved (no swap)
        assert np.isclose(ex / ey, self.PIXDIM[1] / self.PIXDIM[0], rtol=0.05)

    def test_without_pixdim_sphere_is_distorted(self, sphere_volume):
        """Without pixdim the rotation happens in normalized coordinates, so the
        *voxel* extents swap and the object is no longer physically round."""
        aug = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=False)
        out, _ = aug.apply(sphere_volume, aug_spec_list=[self._rot90_spec()])
        ex_in, ey_in, _ = self._extents(sphere_volume)
        ex, ey, _ = self._extents(out)
        # voxel extents swapped
        assert np.isclose(ex, ey_in, rtol=0.05)
        assert np.isclose(ey, ex_in, rtol=0.05)
        # so the physical extents are now badly mismatched
        ratio = (ey * self.PIXDIM[1]) / (ex * self.PIXDIM[0])
        assert ratio > 3.0

    def test_pixdim_changes_the_result(self, sphere_volume):
        spec = self._rot90_spec()
        plain = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=False)
        aniso = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=False,
                                     pixdim=self.PIXDIM)
        out_plain, _ = plain.apply(sphere_volume, aug_spec_list=[spec])
        out_aniso, _ = aniso.apply(sphere_volume, aug_spec_list=[spec])
        assert not torch.allclose(out_plain, out_aniso, atol=1e-3)

    def test_isotropic_pixdim_is_a_no_op(self):
        """Equal physical axis lengths => the correction is the identity."""
        x = torch.randn(1, 16, 16, 16, 2, dtype=torch.float32)
        spec = make_spec(do_z_rot=True, z_angle_deg=17.0)
        plain = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=False)
        iso = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=False,
                                   pixdim=(2.0, 2.0, 2.0))
        out_plain, _ = plain.apply(x, aug_spec_list=[spec])
        out_iso, _ = iso.apply(x, aug_spec_list=[spec])
        assert torch.allclose(out_plain, out_iso, atol=1e-5)

    def test_rot90_on_non_square_fov_warns(self):
        """allow_rot90=True + anisotropic FOV: the user gets told content is lost."""
        aug = SpatialAugmentations(dim=3, prob=1.0, allow_rot90=True,
                                   pixdim=self.PIXDIM)
        x = torch.randn(1, 16, 16, 4, 2, dtype=torch.float32)
        spec = make_spec(do_rot90=True, k_rot90=1)
        with pytest.warns(RuntimeWarning, match='90 degree rotation'):
            aug.apply(x, aug_spec_list=[spec])


#**************************************************************************************************#
#                                    Class TestInputValidation                                     #
#**************************************************************************************************#
#                                                                                                  #
# apply() rejects tensors of the wrong rank with a helpful message.                                #
#                                                                                                  #
#**************************************************************************************************#
class TestInputValidation:
    """apply() rejects tensors of the wrong rank with a helpful message."""

    def test_3d_module_rejects_4d(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        with pytest.raises(ValueError) as excinfo:
            aug.apply(torch.randn(2, 8, 8, 4))
        msg = str(excinfo.value)
        assert '5-D' in msg
        assert 'NIfTI layout' in msg
        assert '(batch, X, Y, Z, channels)' in msg
        assert '(2, 8, 8, 4)' in msg

    def test_2d_module_rejects_5d(self):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        with pytest.raises(ValueError) as excinfo:
            aug.apply(torch.randn(2, 8, 8, 4, 3))
        msg = str(excinfo.value)
        assert '4-D' in msg
        assert '(batch, X, Y, channels)' in msg

    def test_missing_batch_axis_rejected(self):
        """A bare (X, Y, Z, C) volume without a batch axis is a common slip."""
        aug = SpatialAugmentations(dim=3, prob=1.0)
        with pytest.raises(ValueError, match='NIfTI layout'):
            aug.apply(torch.randn(8, 8, 4, 16))

    def test_error_raised_before_sampling(self):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        with pytest.raises(ValueError):
            aug.apply(torch.randn(2, 8, 8))
        assert aug.aug_specs_ is None


#**************************************************************************************************#
#                                    Class TestCoilSubsampling                                     #
#**************************************************************************************************#
#                                                                                                  #
# Coil subsampling happens along the LAST axis in the NIfTI layout.                                #
#                                                                                                  #
#**************************************************************************************************#
class TestCoilSubsampling:
    """Coil subsampling happens along the LAST axis in the NIfTI layout."""

    def test_coil_subsample_reduces_last_axis(self, csm_3d):
        aug = SpatialAugmentations(dim=3, prob=1.0, min_coils=2, max_coils=4)
        out, specs = aug.apply(csm_3d, pipeline='csm')
        if isinstance(out, torch.Tensor):
            assert out.shape[-1] <= csm_3d.shape[-1]
            assert out.shape[:-1] == csm_3d.shape[:-1]  # spatial axes untouched
        else:  # list (varying coil counts)
            for s in out:
                assert s.shape[-1] <= csm_3d.shape[-1]

    def test_coil_keep_is_respected(self, csm_3d):
        aug = SpatialAugmentations(dim=3, prob=1.0, min_coils=3, max_coils=3)
        out, specs = aug.apply(csm_3d, pipeline='csm')
        assert all(s['do_coil_sub'] and s['coil_keep'] == 3 for s in specs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (csm_3d.shape[0], *csm_3d.shape[1:-1], 3)

    def test_no_coil_subsample_in_data_pipeline(self, csm_3d):
        aug = SpatialAugmentations(dim=3, prob=1.0, min_coils=2, max_coils=4)
        out, _ = aug.apply(csm_3d, pipeline='data')
        assert isinstance(out, torch.Tensor)
        assert out.shape[-1] == csm_3d.shape[-1]

    def test_default_coil_sampler_uses_last_axis(self):
        aug = SpatialAugmentations()
        x = torch.randn(1, 8, 8, 4, 6)
        out = aug.default_coil_sampler(x, keep=2)
        assert out.shape == (1, 8, 8, 4, 2)

    def test_default_coil_sampler_clamps_to_available(self):
        aug = SpatialAugmentations()
        x = torch.randn(1, 8, 8, 4, 6)
        assert aug.default_coil_sampler(x, keep=99).shape[-1] == 6
        assert aug.default_coil_sampler(x, keep=0).shape[-1] == 1

    def test_default_coil_sampler_none_keep(self):
        aug = SpatialAugmentations()
        x = torch.randn(1, 8, 8, 4, 6)
        out = aug.default_coil_sampler(x, keep=None)
        assert out.shape == x.shape

    def test_default_coil_sampler_explicit_axis(self):
        aug = SpatialAugmentations()
        x = torch.randn(1, 6, 8, 8, 4)
        out = aug.default_coil_sampler(x, keep=2, coils_axis=1)
        assert out.shape == (1, 2, 8, 8, 4)


#**************************************************************************************************#
#                                     Class TestProcessTensor                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test process_tensor (BaseModule interface).                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestProcessTensor:
    """Test process_tensor (BaseModule interface)."""

    def test_numpy_roundtrip(self, spatial_3d):
        arr = np.random.randn(2, 8, 8, 4, 16).astype(np.float32)
        out, water = spatial_3d.process_tensor(arr)
        assert isinstance(out, np.ndarray)
        assert out.shape == arr.shape
        assert water is None

    def test_numpy_complex_roundtrip(self, spatial_3d):
        arr = (np.random.randn(2, 8, 8, 4, 16)
               + 1j * np.random.randn(2, 8, 8, 4, 16)).astype(np.complex64)
        out, _ = spatial_3d.process_tensor(arr)
        assert isinstance(out, np.ndarray)
        assert np.iscomplexobj(out)
        assert out.shape == arr.shape

    def test_numpy_roundtrip_2d(self):
        aug = SpatialAugmentations(dim=2, prob=1.0)
        arr = np.random.randn(3, 16, 16, 8).astype(np.float32)
        out, _ = aug.process_tensor(arr)
        assert isinstance(out, np.ndarray)
        assert out.shape == arr.shape

    def test_torch_input(self, spatial_3d, batch_3d_real):
        out, _ = spatial_3d.process_tensor(batch_3d_real)
        assert isinstance(out, torch.Tensor)
        assert out.shape == batch_3d_real.shape

    def test_stores_aug_specs(self, spatial_3d, batch_3d_real):
        spatial_3d.process_tensor(batch_3d_real)
        assert spatial_3d.aug_specs_ is not None
        assert len(spatial_3d.aug_specs_) == batch_3d_real.shape[0]

    def test_water_passthrough(self, spatial_3d, batch_3d_real):
        water = torch.randn_like(batch_3d_real)
        _, water_out = spatial_3d.process_tensor(batch_3d_real, water_array=water)
        assert water_out is water  # unchanged, same object

    def test_replay_via_process_tensor(self, spatial_3d):
        arr = np.random.randn(2, 8, 8, 4, 16).astype(np.float32)
        out_a, _ = spatial_3d.process_tensor(arr)
        specs = spatial_3d.aug_specs_
        out_b, _ = spatial_3d.process_tensor(arr, aug_spec_list=specs)
        assert np.array_equal(out_a, out_b)


#**************************************************************************************************#
#                                    Class TestProcessNiftiList                                    #
#**************************************************************************************************#
#                                                                                                  #
# Test process_nifti_list (BaseModule interface) with real NIfTI objects.                          #
#                                                                                                  #
#**************************************************************************************************#
class TestProcessNiftiList:
    """Test process_nifti_list (BaseModule interface) with real NIfTI objects."""

    def test_returns_correct_length(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        result_data, result_water = aug(nifti_plus, None)
        assert len(result_data) == len(spatial_nifti_list)
        assert result_water is None

    def test_shape_preserved(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        assert nifti_plus.shape == (3, 8, 8, 4, 16)
        result_data, _ = aug(nifti_plus, None)
        assert result_data.shape == (3, 8, 8, 4, 16)

    def test_data_changes(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        original = nifti_plus[0][:].copy()
        result_data, _ = aug(nifti_plus, None)
        augmented = result_data[0][:]
        assert not np.allclose(augmented, original)

    def test_identity_at_prob_zero(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=0.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        original = nifti_plus[0][:].copy()
        result_data, _ = aug(nifti_plus, None)
        assert np.array_equal(result_data[0][:], original)

    def test_stores_aug_specs(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        aug(nifti_plus, None)
        assert aug.aug_specs_ is not None
        assert len(aug.aug_specs_) == len(spatial_nifti_list)

    def test_water_passthrough(self, spatial_nifti_list, spatial_nifti_water):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        data_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                   backend=Backend.NIFTI_LIST)
        water_plus = NIfTI_MRS_Plus(nifti_list=[spatial_nifti_water],
                                    backend=Backend.NIFTI_LIST)
        _, result_water = aug(data_plus, water_plus)
        # Water should be returned unchanged
        assert result_water is not None


#**************************************************************************************************#
#                               Class TestSpatialPipelineIntegration                               #
#**************************************************************************************************#
#                                                                                                  #
# Test SpatialAugmentations inside an AugmentationPipeline.                                        #
#                                                                                                  #
#**************************************************************************************************#
class TestSpatialPipelineIntegration:
    """Test SpatialAugmentations inside an AugmentationPipeline."""

    def test_in_pipeline_alone(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        pipeline = AugmentationPipeline([aug])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)
        assert len(result_data) == len(spatial_nifti_list)
        assert result_data.shape == (3, 8, 8, 4, 16)

    def test_in_pipeline_with_noise(self, spatial_nifti_list):
        from augmentrum.augmentation.gaussian_noise import GaussianNoise
        aug = SpatialAugmentations(dim=3, prob=0.5)
        noise = GaussianNoise(sigma_frac=0.01)
        pipeline = AugmentationPipeline([aug, noise])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)
        assert len(result_data) == len(spatial_nifti_list)

    def test_batch_params_injection(self, spatial_nifti_list):
        """Verify the pipeline can inject sampled scalars via batch_params."""
        aug = SpatialAugmentations(dim=3, prob=1.0, max_z_angle_deg=10.0)
        pipeline = AugmentationPipeline([aug])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)

        # Simulate what the pipeline does in on-the-fly mode:
        # inject a different max_z_angle_deg for this batch
        batch_params = {0: {'max_z_angle_deg': 90.0}}
        result_data, _ = pipeline(data=nifti_plus, water=None,
                                  batch_params=batch_params)
        assert len(result_data) == len(spatial_nifti_list)
        # After call, original value should be restored
        assert aug.max_z_angle_deg == 10.0

    def test_zoom_min_max_injection(self, spatial_nifti_list):
        """Pipeline can inject zoom_min/zoom_max as sampled scalars."""
        aug = SpatialAugmentations(dim=3, prob=1.0, zoom_min=0.9, zoom_max=1.1)
        pipeline = AugmentationPipeline([aug])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)

        batch_params = {0: {'zoom_min': 0.5, 'zoom_max': 0.6}}
        result_data, _ = pipeline(data=nifti_plus, water=None,
                                  batch_params=batch_params)
        assert len(result_data) == len(spatial_nifti_list)
        # Originals restored
        assert aug.zoom_min == 0.9
        assert aug.zoom_max == 1.1

    def test_translation_prob_injection(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0, translation_prob=0.1)
        pipeline = AugmentationPipeline([aug])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list,
                                    backend=Backend.NIFTI_LIST)
        batch_params = {0: {'translation_prob': 1.0}}
        result_data, _ = pipeline(data=nifti_plus, water=None,
                                  batch_params=batch_params)
        assert len(result_data) == len(spatial_nifti_list)
        assert aug.translation_prob == 0.1


#**************************************************************************************************#
#                                     Class TestRoutePipeline                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test the convenience route_pipeline method.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class TestRoutePipeline:
    """Test the convenience route_pipeline method."""

    def test_data_only(self, spatial_3d, batch_3d_real):
        results = spatial_3d.route_pipeline(data_list=batch_3d_real)
        assert 'data' in results
        assert results['data'].shape == batch_3d_real.shape
        assert 'data_augmentations' in results

    def test_csm_only(self, csm_3d):
        aug = SpatialAugmentations(dim=3, prob=1.0, min_coils=2, max_coils=4)
        results = aug.route_pipeline(csm_list=csm_3d)
        assert 'csm' in results
        assert 'csm_augmentations' in results

    def test_data_and_csm(self, spatial_3d, batch_3d_real, csm_3d):
        results = spatial_3d.route_pipeline(data_list=batch_3d_real, csm_list=csm_3d)
        assert 'data' in results
        assert 'csm' in results

    def test_numpy_data(self, spatial_3d):
        arr = np.random.randn(2, 8, 8, 4, 16).astype(np.float32)
        results = spatial_3d.route_pipeline(data_list=arr)
        assert isinstance(results['data'], np.ndarray)
        assert results['data'].shape == arr.shape

    def test_precomputed_specs_are_reused(self, spatial_3d, batch_3d_real):
        specs = spatial_3d.sample_augmentations(batch_3d_real.shape[0])
        r1 = spatial_3d.route_pipeline(data_list=batch_3d_real, data_aug_list=specs)
        r2 = spatial_3d.route_pipeline(data_list=batch_3d_real, data_aug_list=specs)
        assert torch.equal(r1['data'], r2['data'])
        assert r1['data_augmentations'] is specs


#**************************************************************************************************#
#                                 Class TestAugmentrumRegistration                                 #
#**************************************************************************************************#
#                                                                                                  #
# Verify SpatialAugmentations is discoverable from Augmentrum.                                     #
#                                                                                                  #
#**************************************************************************************************#
class TestAugmentrumRegistration:
    """Verify SpatialAugmentations is discoverable from Augmentrum."""

    def test_available_modules(self):
        from augmentrum.core.augmentrum import Augmentrum
        assert 'spatial' in Augmentrum.AVAILABLE_MODULES
        assert 'spatial_augmentations' in Augmentrum.AVAILABLE_MODULES
        assert Augmentrum.AVAILABLE_MODULES['spatial'] is SpatialAugmentations

    def test_importable_from_package(self):
        from augmentrum.augmentation import SpatialAugmentations as SA
        assert SA is SpatialAugmentations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
