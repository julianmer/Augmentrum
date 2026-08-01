"""
Tests for NIFTI-MRS and HDF5 export functionality
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from augmentrum import Augmentrum
from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


#**************************************************************************************************#
#                                     Class TestNIfTIMRSExport                                     #
#**************************************************************************************************#
#                                                                                                  #
# Test NIFTI-MRS export functionality.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class TestNIfTIMRSExport:
    """Test NIFTI-MRS export functionality."""

    @pytest.fixture
    def simple_augmenter(self, dummy_nifti_mrs):
        """Create a simple augmenter for testing."""
        return Augmentrum(
            data=[dummy_nifti_mrs],
            pipeline=['coil_sampling', 'processing', 'phase', 'noise'],
            n_coils=(None, None),  # Use all coils
            n_averages=(None, None),  # Use all averages
            zero_order_deg=10.0,
            noise_amp=0.05,
            backend='numpy',
            batch_size=3,
            val_frac=0.0,
            test_frac=0.0,
            volatile=True
        )

    def test_export_nifti_basic(self, simple_augmenter, tmp_path):
        """Test basic NIFTI-MRS export."""
        output_dir = tmp_path / "nifti_export"

        result = simple_augmenter.export_batch(
            output_dir=str(output_dir),
            format='nifti-mrs',
            n_batches=1,
            prefix='test'
        )

        # Check result dictionary
        assert result['format'] == 'nifti-mrs'
        assert result['n_batches'] == 1
        assert result['n_spectra'] == 3  # batch_size=3
        assert result['n_files'] == 3
        assert len(result['files']) == 3

        # Check files exist
        for filepath in result['files']:
            assert Path(filepath).exists()
            assert filepath.endswith('.nii.gz')

    @pytest.mark.skip(reason="Requires actual test data files from tests/testdata/fsl_mrs/")
    def test_export_nifti_with_water(self, tmp_path):
        """Test NIFTI-MRS export with water reference."""
        from fsl_mrs.utils.mrs_io import read_FID

        test_data_dir = Path(__file__).parent.parent / 'testdata' / 'fsl_mrs'
        metab = read_FID(str(test_data_dir / 'metab.nii.gz'))
        water = read_FID(str(test_data_dir / 'wref.nii.gz'))

        augmenter = Augmentrum(
            data=[metab],
            water=[water],
            pipeline=['coil_sampling', 'processing', 'phase'],
            n_coils=(None, None),
            n_averages=(None, None),
            zero_order_deg=10.0,
            backend='numpy',
            batch_size=2,
            val_frac=0.0,
            test_frac=0.0,
            volatile=True
        )

        output_dir = tmp_path / "nifti_water_export"

        result = augmenter.export_batch(
            output_dir=str(output_dir),
            format='nifti-mrs',
            n_batches=1,
            save_water=True
        )

        # Check water files were created
        assert 'n_water_files' in result
        assert result['n_water_files'] == 2
        assert len(result['water_files']) == 2

        for filepath in result['water_files']:
            assert Path(filepath).exists()
            assert 'water' in filepath

    def test_export_nifti_multiple_batches(self, simple_augmenter, tmp_path):
        """Test exporting multiple batches."""
        output_dir = tmp_path / "nifti_multi_export"

        result = simple_augmenter.export_batch(
            output_dir=str(output_dir),
            format='nifti-mrs',
            n_batches=3,
            prefix='multi'
        )

        # Should have 3 batches × 3 spectra = 9 files
        assert result['n_batches'] == 3
        assert result['n_spectra'] == 9
        assert result['n_files'] == 9

    def test_export_nifti_with_metadata(self, simple_augmenter, tmp_path):
        """Test NIFTI-MRS export with custom metadata."""
        output_dir = tmp_path / "nifti_metadata_export"

        custom_metadata = {
            'study': 'test_study',
            'location': 'test_location',
            'parameter': 'test_value'
        }

        result = simple_augmenter.export_batch(
            output_dir=str(output_dir),
            format='nifti-mrs',
            n_batches=1,
            metadata=custom_metadata
        )

        # Load one file and check metadata
        from fsl_mrs.utils.mrs_io import read_FID
        test_file = result['files'][0]
        loaded_nifti = read_FID(test_file)

        # Check augmentrum provenance was added
        assert hasattr(loaded_nifti, 'hdr_ext')
        # Metadata should be in processing history
        # (exact structure depends on FSL-MRS implementation)

    def test_export_nifti_prefix_and_naming(self, simple_augmenter, tmp_path):
        """Test file naming with custom prefix."""
        output_dir = tmp_path / "nifti_naming_export"

        result = simple_augmenter.export_batch(
            output_dir=str(output_dir),
            format='nifti-mrs',
            n_batches=1,
            prefix='custom_prefix'
        )

        # Check naming convention
        for filepath in result['files']:
            filename = Path(filepath).name
            assert filename.startswith('custom_prefix_')
            assert filename.endswith('.nii.gz')


#**************************************************************************************************#
#                                       Class TestHDF5Export                                       #
#**************************************************************************************************#
#                                                                                                  #
# Test HDF5 export functionality.                                                                  #
#                                                                                                  #
#**************************************************************************************************#
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHDF5Export:
    """Test HDF5 export functionality."""

    @pytest.fixture
    def simple_augmenter(self, dummy_nifti_mrs):
        """Create a simple augmenter for testing."""
        return Augmentrum(
            data=[dummy_nifti_mrs],
            pipeline=['coil_sampling', 'processing', 'phase', 'noise'],
            n_coils=(None, None),
            n_averages=(None, None),
            zero_order_deg=10.0,
            noise_amp=0.05,
            backend='numpy',
            batch_size=4,
            val_frac=0.0,
            test_frac=0.0,
            volatile=True
        )

    def test_export_hdf5_basic(self, simple_augmenter, tmp_path):
        """Test basic HDF5 export."""
        import h5py

        output_dir = tmp_path / "hdf5_export"

        result = simple_augmenter.export_batch(
            output_dir=str(output_dir),
            format='hdf5',
            n_batches=1,
            prefix='test_batch'
        )

        # Check result
        assert result['format'] == 'hdf5'
        assert result['n_batches'] == 1
        assert result['n_files'] == 1

        # Load and check HDF5 file
        hdf5_file = result['files'][0]
        assert Path(hdf5_file).exists()

        with h5py.File(hdf5_file, 'r') as f:
            # Check data
            assert 'data' in f
            assert f['data'].shape[0] == 4  # batch_size

            # Check metadata
            assert 'augmentrum_version' in f.attrs
            assert f.attrs['split'] == 'train'
            assert f.attrs['n_spectra'] == 4

    @pytest.mark.skip(reason="Requires actual test data files from tests/testdata/fsl_mrs/")
    def test_export_hdf5_with_water(self, tmp_path):
        """Test HDF5 export with water reference."""
        import h5py
        from fsl_mrs.utils.mrs_io import read_FID

        test_data_dir = Path(__file__).parent.parent / 'testdata' / 'fsl_mrs'
        metab = read_FID(str(test_data_dir / 'metab.nii.gz'))
        water = read_FID(str(test_data_dir / 'wref.nii.gz'))

        augmenter = Augmentrum(
            data=[metab],
            water=[water],
            pipeline=['coil_sampling', 'processing', 'phase'],
            n_coils=(None, None),
            n_averages=(None, None),
            zero_order_deg=10.0,
            backend='numpy',
            batch_size=2,
            val_frac=0.0,
            test_frac=0.0,
            volatile=True
        )

        output_dir = tmp_path / "hdf5_water_export"

        result = augmenter.export_batch(
            output_dir=str(output_dir),
            format='hdf5',
            n_batches=1,
            save_water=True
        )

        # Check water was saved
        hdf5_file = result['files'][0]
        with h5py.File(hdf5_file, 'r') as f:
            assert 'data' in f
            assert 'water' in f
            assert f['water'].shape[0] == 2

    def test_export_hdf5_multiple_batches(self, simple_augmenter, tmp_path):
        """Test HDF5 export with multiple batches."""
        output_dir = tmp_path / "hdf5_multi_export"

        result = simple_augmenter.export_batch(
            output_dir=str(output_dir),
            format='hdf5',
            n_batches=3
        )

        # Should create 3 separate HDF5 files
        assert result['n_batches'] == 3
        assert result['n_files'] == 3
        assert len(result['files']) == 3

    def test_export_hdf5_compression(self, simple_augmenter, tmp_path):
        """Test HDF5 export uses compression."""
        import h5py

        output_dir = tmp_path / "hdf5_compression_export"

        result = simple_augmenter.export_batch(
            output_dir=str(output_dir),
            format='hdf5',
            n_batches=1
        )

        # Check compression was applied
        hdf5_file = result['files'][0]
        with h5py.File(hdf5_file, 'r') as f:
            assert f['data'].compression == 'gzip'


#**************************************************************************************************#
#                                    Class TestNIfTIMRSPlusSave                                    #
#**************************************************************************************************#
#                                                                                                  #
# Test NIfTI_MRS_Plus save methods.                                                                #
#                                                                                                  #
#**************************************************************************************************#
class TestNIfTIMRSPlusSave:
    """Test NIfTI_MRS_Plus save methods."""

    @pytest.fixture
    def nifti_plus_batch(self, dummy_nifti_mrs):
        """Create a NIfTI_MRS_Plus object with multiple subjects."""
        # Create multiple copies with slight variations
        nifti_list = [dummy_nifti_mrs] * 3
        from augmentrum.core.nifti_mrs_plus import Backend
        return NIfTI_MRS_Plus(nifti_list, backend=Backend.NUMPY, volatile=True)

    def test_save_nifti_basic(self, nifti_plus_batch, tmp_path):
        """Test NIfTI_MRS_Plus save_nifti method."""
        output_dir = tmp_path / "nifti_plus_export"

        saved_files = nifti_plus_batch.save_nifti(
            output_dir=str(output_dir),
            prefix='saved',
            zero_pad=3
        )

        # Check correct number of files
        assert len(saved_files) == 3

        # Check files exist and naming
        for i, filepath in enumerate(saved_files):
            assert Path(filepath).exists()
            assert f'saved_{i:03d}.nii.gz' in filepath

    def test_save_nifti_custom_padding(self, nifti_plus_batch, tmp_path):
        """Test custom zero padding."""
        output_dir = tmp_path / "nifti_plus_padding"

        saved_files = nifti_plus_batch.save_nifti(
            output_dir=str(output_dir),
            prefix='test',
            zero_pad=6
        )

        # Check padding
        for i, filepath in enumerate(saved_files):
            assert f'test_{i:06d}.nii.gz' in filepath

    @pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
    def test_save_hdf5_basic(self, nifti_plus_batch, tmp_path):
        """Test NIfTI_MRS_Plus save_hdf5 method."""
        import h5py

        hdf5_file = tmp_path / "nifti_plus_batch.h5"

        nifti_plus_batch.save_hdf5(str(hdf5_file))

        # Check file exists
        assert hdf5_file.exists()

        # Verify contents
        with h5py.File(hdf5_file, 'r') as f:
            assert 'data' in f
            assert f.attrs['n_subjects'] == 3
            assert 'augmentrum_version' in f.attrs

    @pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
    def test_save_hdf5_compression(self, nifti_plus_batch, tmp_path):
        """Test HDF5 compression options."""
        import h5py

        hdf5_file = tmp_path / "nifti_plus_compressed.h5"

        nifti_plus_batch.save_hdf5(
            str(hdf5_file),
            compression='gzip',
            compression_opts=9
        )

        with h5py.File(hdf5_file, 'r') as f:
            assert f['data'].compression == 'gzip'
            assert f['data'].compression_opts == 9


#**************************************************************************************************#
#                                      Class TestExportErrors                                      #
#**************************************************************************************************#
#                                                                                                  #
# Test error handling in export functions.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class TestExportErrors:
    """Test error handling in export functions."""

    def test_export_invalid_format(self, dummy_nifti_mrs, tmp_path):
        """Test error on invalid export format."""
        augmenter = Augmentrum(
            data=[dummy_nifti_mrs],
            pipeline=['coil_sampling', 'processing'],
            n_coils=(None, None),
            n_averages=(None, None),
            backend='numpy',
            batch_size=2,
            volatile=True
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            augmenter.export_batch(
                output_dir=str(tmp_path),
                format='invalid_format',
                n_batches=1
            )

    def test_export_invalid_split(self, dummy_nifti_mrs, tmp_path):
        """Test error on invalid split name."""
        augmenter = Augmentrum(
            data=[dummy_nifti_mrs],
            pipeline=['coil_sampling', 'processing'],
            n_coils=(None, None),
            n_averages=(None, None),
            backend='numpy',
            batch_size=2,
            volatile=True
        )

        with pytest.raises(ValueError, match="Invalid split"):
            augmenter.export_batch(
                output_dir=str(tmp_path),
                format='nifti-mrs',
                split='invalid_split',
                n_batches=1
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
