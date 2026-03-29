"""
Tests for SpatialAugmentations module.

Tests cover:
- Creation / initialization
- Sampling augmentation specs
- 2D and 3D affine application
- Complex data handling
- Coil subsampling
- NIfTI list processing (BaseModule interface)
- Tensor processing (BaseModule interface)
- Pipeline integration
- Backend compatibility
- Route-pipeline convenience method
- Parameter injection from Augmentrum (zoom_min / zoom_max / etc.)
"""

import pytest
import numpy as np
import torch

from augmentrum.augmentation.spatial_augmentations import SpatialAugmentations
from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend
from augmentrum.core.pipeline import AugmentationPipeline


# ─────────────────────────────────────────────────────────
# Fixtures — spatial NIfTI objects (image-like, not MRS)
# ─────────────────────────────────────────────────────────

@pytest.fixture
def spatial_nifti_list():
    """Create a list of 3 NIfTI-MRS objects with 4-D spatial shape suitable for
    SpatialAugmentations: each item has shape (C, D, H, W) so that stacking
    into a batch gives (N, C, D, H, W).
    """
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    nifti_list = []
    for _ in range(3):
        # shape (1, 8, 8, 8) → after stack → (3, 1, 8, 8, 8) = 5-D ✓
        data = (np.random.randn(1, 8, 8, 8) + 1j * np.random.randn(1, 8, 8, 8)).astype(np.complex64)
        nifti = gen_nifti_mrs(data, 1/2000, 123.0)
        nifti_list.append(nifti)
    return nifti_list


@pytest.fixture
def spatial_nifti_water():
    """Single water-ref NIfTI with matching spatial shape."""
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    data = (np.random.randn(1, 8, 8, 8) + 1j * np.random.randn(1, 8, 8, 8)).astype(np.complex64)
    return gen_nifti_mrs(data, 1/2000, 123.0)

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
    """Real-valued 2-D batch tensor [N, C, H, W]."""
    return torch.randn(4, 1, 32, 32, dtype=torch.float32)


@pytest.fixture
def batch_2d_complex():
    """Complex-valued 2-D batch tensor [N, C, H, W]."""
    return (torch.randn(4, 1, 32, 32) + 1j * torch.randn(4, 1, 32, 32)).to(torch.complex64)


@pytest.fixture
def batch_3d_real():
    """Real-valued 3-D batch tensor [N, C, D, H, W]."""
    return torch.randn(2, 1, 8, 16, 16, dtype=torch.float32)


@pytest.fixture
def batch_3d_complex():
    """Complex-valued 3-D batch tensor [N, C, D, H, W]."""
    return (torch.randn(2, 1, 8, 16, 16) + 1j * torch.randn(2, 1, 8, 16, 16)).to(torch.complex64)


@pytest.fixture
def csm_3d():
    """Multi-coil 3-D complex tensor for CSM pipeline [N, 8, D, H, W]."""
    return (torch.randn(2, 8, 8, 16, 16) + 1j * torch.randn(2, 8, 8, 16, 16)).to(torch.complex64)


# ─────────────────────────────────────────────────────────
# Creation
# ─────────────────────────────────────────────────────────

class TestSpatialCreation:
    """Test SpatialAugmentations initialization."""

    def test_defaults(self):
        aug = SpatialAugmentations()
        assert aug.dim == 3
        assert aug.prob == 0.5
        assert aug.pipeline == 'data'
        assert aug.zoom_min == 0.9
        assert aug.zoom_max == 1.1
        assert aug.scale_min == 0.9
        assert aug.scale_max == 1.1

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

    def test_invalid_dim(self):
        with pytest.raises(AssertionError):
            SpatialAugmentations(dim=4)

    def test_data_ranges_property(self):
        aug = SpatialAugmentations(zoom_min=0.8, zoom_max=1.3, shear_max=0.2)
        r = aug.data_ranges
        assert r['zoom_range'] == (0.8, 1.3)
        assert r['shear_max'] == 0.2

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


# ─────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────

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
            'do_zoom', 'zoom_factor',
            'do_shear', 'shear_xy', 'shear_z',
            'do_flip', 'flip_x', 'flip_y', 'flip_z',
            'do_anisotropic', 'scale_xyz',
            'do_coil_sub', 'coil_keep',
            'pipeline',
        }
        assert required.issubset(spec.keys())

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
            assert not s['do_zoom']
            assert not s['do_shear']
            assert not s['do_flip']

    def test_prob_one_enables_all(self):
        aug = SpatialAugmentations(prob=1.0)
        specs = aug.sample_augmentations(50)
        for s in specs:
            assert s['do_translate']
            assert s['do_z_rot']
            assert s['do_zoom']
            assert s['do_shear']
            assert s['do_flip']

    def test_zoom_factor_within_range(self):
        aug = SpatialAugmentations(prob=1.0, zoom_min=0.8, zoom_max=1.2)
        specs = aug.sample_augmentations(100)
        for s in specs:
            assert 0.8 <= s['zoom_factor'] <= 1.2

    def test_scale_within_range(self):
        aug = SpatialAugmentations(prob=1.0, scale_min=0.7, scale_max=1.3)
        specs = aug.sample_augmentations(100)
        for s in specs:
            sx, sy, sz = s['scale_xyz']
            assert 0.7 <= sx <= 1.3
            assert 0.7 <= sy <= 1.3
            assert 0.7 <= sz <= 1.3


# ─────────────────────────────────────────────────────────
# Affine application — 2D
# ─────────────────────────────────────────────────────────

class TestSpatialApply2D:
    """Test apply() on 2-D data."""

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
        assert torch.allclose(out, batch_2d_real, atol=1e-5)


# ─────────────────────────────────────────────────────────
# Affine application — 3D
# ─────────────────────────────────────────────────────────

class TestSpatialApply3D:
    """Test apply() on 3-D data."""

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
        assert torch.allclose(out, batch_3d_real, atol=1e-5)


# ─────────────────────────────────────────────────────────
# Coil subsampling
# ──────────────────────────────────────────���──────────────

class TestCoilSubsampling:
    """Test coil subsampling via CSM pipeline."""

    def test_coil_subsample_reduces_channels(self, csm_3d):
        aug = SpatialAugmentations(dim=3, prob=1.0, min_coils=2, max_coils=4)
        out, specs = aug.apply(csm_3d, pipeline='csm')
        # Output should have fewer coils when subsampled
        if isinstance(out, torch.Tensor):
            assert out.shape[1] <= csm_3d.shape[1]
        else:  # list (varying coil counts)
            for s in out:
                assert s.shape[1] <= csm_3d.shape[1]

    def test_no_coil_subsample_in_data_pipeline(self, csm_3d):
        aug = SpatialAugmentations(dim=3, prob=1.0, min_coils=2, max_coils=4)
        out, _ = aug.apply(csm_3d, pipeline='data')
        # data pipeline should NOT subsample coils
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == csm_3d.shape[1]

    def test_default_coil_sampler_clamps(self):
        aug = SpatialAugmentations()
        x = torch.randn(1, 4, 8, 8, 8)
        out = aug.default_coil_sampler(x, keep=2)
        assert out.shape[1] == 2

    def test_default_coil_sampler_none_keep(self):
        aug = SpatialAugmentations()
        x = torch.randn(1, 4, 8, 8, 8)
        out = aug.default_coil_sampler(x, keep=None)
        assert out.shape == x.shape


# ─────────────────────────────────────────────────────────
# BaseModule — process_tensor
# ─────────────────────────────────────────────────────────

class TestProcessTensor:
    """Test process_tensor (BaseModule interface)."""

    def test_numpy_roundtrip(self, spatial_3d):
        arr = np.random.randn(2, 1, 8, 16, 16).astype(np.float32)
        out, water = spatial_3d.process_tensor(arr)
        assert isinstance(out, np.ndarray)
        assert out.shape == arr.shape
        assert water is None

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


# ─────────────────────────────────────────────────────────
# BaseModule — process_nifti_list
# ─────────────────────────────────────────────────────────

class TestProcessNiftiList:
    """Test process_nifti_list (BaseModule interface) with real NIfTI objects."""

    def test_returns_correct_length(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, result_water = aug(nifti_plus, None)
        assert len(result_data) == len(spatial_nifti_list)
        assert result_water is None

    def test_data_changes(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)
        original = nifti_plus[0][:].copy()
        result_data, _ = aug(nifti_plus, None)
        augmented = result_data[0][:]
        assert not np.allclose(augmented, original)

    def test_stores_aug_specs(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)
        aug(nifti_plus, None)
        assert aug.aug_specs_ is not None
        assert len(aug.aug_specs_) == len(spatial_nifti_list)

    def test_water_passthrough(self, spatial_nifti_list, spatial_nifti_water):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        data_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)
        water_plus = NIfTI_MRS_Plus(nifti_list=[spatial_nifti_water], backend=Backend.NIFTI_LIST)
        _, result_water = aug(data_plus, water_plus)
        # Water should be returned unchanged
        assert result_water is not None


# ─────────────────────────────────────────────────────────
# Pipeline integration
# ─────────────────────────────────────────────────────────

class TestSpatialPipelineIntegration:
    """Test SpatialAugmentations inside an AugmentationPipeline."""

    def test_in_pipeline_alone(self, spatial_nifti_list):
        aug = SpatialAugmentations(dim=3, prob=1.0)
        pipeline = AugmentationPipeline([aug])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)
        assert len(result_data) == len(spatial_nifti_list)

    def test_in_pipeline_with_noise(self, spatial_nifti_list):
        from augmentrum.augmentation.gaussian_noise import GaussianNoise
        aug = SpatialAugmentations(dim=3, prob=0.5)
        noise = GaussianNoise(sigma_frac=0.01)
        pipeline = AugmentationPipeline([aug, noise])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)
        result_data, _ = pipeline(data=nifti_plus, water=None)
        assert len(result_data) == len(spatial_nifti_list)

    def test_batch_params_injection(self, spatial_nifti_list):
        """Verify the pipeline can inject sampled scalars via batch_params."""
        aug = SpatialAugmentations(dim=3, prob=1.0, max_z_angle_deg=10.0)
        pipeline = AugmentationPipeline([aug])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)

        # Simulate what the pipeline does in on-the-fly mode:
        # inject a different max_z_angle_deg for this batch
        batch_params = {0: {'max_z_angle_deg': 90.0}}
        result_data, _ = pipeline(data=nifti_plus, water=None, batch_params=batch_params)
        assert len(result_data) == len(spatial_nifti_list)
        # After call, original value should be restored
        assert aug.max_z_angle_deg == 10.0

    def test_zoom_min_max_injection(self, spatial_nifti_list):
        """Pipeline can inject zoom_min/zoom_max as sampled scalars."""
        aug = SpatialAugmentations(dim=3, prob=1.0, zoom_min=0.9, zoom_max=1.1)
        pipeline = AugmentationPipeline([aug])
        nifti_plus = NIfTI_MRS_Plus(nifti_list=spatial_nifti_list, backend=Backend.NIFTI_LIST)

        batch_params = {0: {'zoom_min': 0.5, 'zoom_max': 0.6}}
        result_data, _ = pipeline(data=nifti_plus, water=None, batch_params=batch_params)
        assert len(result_data) == len(spatial_nifti_list)
        # Originals restored
        assert aug.zoom_min == 0.9
        assert aug.zoom_max == 1.1


# ─────────────────────────────────────────────────────────
# Route pipeline
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# Augmentrum-level registration
# ─────────────────────────────────────────────────────────

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




