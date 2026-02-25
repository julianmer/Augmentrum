"""
Test the pipeline visualization feature.
"""

import pytest
from augmentrum import Augmentrum


class TestPipelineVisualization:
    """Test pipeline visualization methods."""

    def test_visualize_pipeline_returns_string(self, dummy_nifti_list):
        """Test that visualize_pipeline returns a string."""
        augmenter = Augmentrum(data=dummy_nifti_list, pipeline=['noise'])
        viz = augmenter.visualize_pipeline()

        assert isinstance(viz, str)
        assert len(viz) > 0

    def test_visualization_contains_header(self, dummy_nifti_list):
        """Test that visualization contains the header."""
        augmenter = Augmentrum(data=dummy_nifti_list)
        viz = augmenter.visualize_pipeline()

        assert 'AUGMENTRUM PIPELINE' in viz
        assert '═' in viz  # Box drawing characters

    def test_visualization_shows_data_info(self, dummy_nifti_list):
        """Test that visualization shows data information."""
        augmenter = Augmentrum(data=dummy_nifti_list, batch_size=8)
        viz = augmenter.visualize_pipeline()

        assert 'subjects' in viz.lower()
        assert 'Backend' in viz
        assert 'Batch Size' in viz
        assert '8' in viz

    def test_visualization_shows_water_info(self, dummy_nifti_list):
        """Test that visualization shows water reference info."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            water=dummy_nifti_list[:2]
        )
        viz = augmenter.visualize_pipeline()

        assert 'Water' in viz
        assert '2' in viz

    def test_visualization_shows_pipeline_steps(self, dummy_nifti_list):
        """Test that visualization shows pipeline steps."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise', 'line_broadening']
        )
        viz = augmenter.visualize_pipeline()

        assert 'Pipeline Steps' in viz
        assert 'GaussianNoise' in viz
        assert 'LineBroadening' in viz

    def test_visualization_detailed_shows_params(self, dummy_nifti_list):
        """Test that detailed visualization shows parameters."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma_frac=0.05  # User param (stored in Pipeline, not shown in viz)
        )
        viz = augmenter.visualize_pipeline(detailed=True)

        assert 'sigma_frac' in viz
        # Module is created with default 0.02, user param 0.05 is in Pipeline
        assert '0.02' in viz or '0.020' in viz

    def test_visualization_simple_hides_params(self, dummy_nifti_list):
        """Test that simple visualization hides parameters."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['noise'],
            sigma_frac=0.05
        )
        viz = augmenter.visualize_pipeline(detailed=False)

        # Should show module name but not parameters
        assert 'GaussianNoise' in viz
        assert 'sigma_frac' not in viz

    def test_show_pipeline_prints(self, dummy_nifti_list, capsys):
        """Test that show_pipeline prints to stdout."""
        augmenter = Augmentrum(data=dummy_nifti_list, pipeline=['noise'])
        augmenter.show_pipeline()

        captured = capsys.readouterr()
        assert 'AUGMENTRUM PIPELINE' in captured.out
        assert 'GaussianNoise' in captured.out

    def test_empty_pipeline_visualization(self, dummy_nifti_list):
        """Test visualization with empty pipeline."""
        augmenter = Augmentrum(data=dummy_nifti_list, pipeline=[])
        viz = augmenter.visualize_pipeline()

        assert 'No augmentation modules' in viz

    def test_visualization_has_emojis(self, dummy_nifti_list):
        """Test that visualization includes emojis."""
        augmenter = Augmentrum(
            data=dummy_nifti_list,
            pipeline=['processing', 'noise', 'line_broadening', 'baseline']
        )
        viz = augmenter.visualize_pipeline()

        # Should have various emojis
        assert '🔬' in viz  # Header
        assert '📊' in viz  # Data
        assert '🎯' in viz  # Backend
        assert '⚙️' in viz or '📡' in viz or '〰️' in viz  # Module emojis


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
