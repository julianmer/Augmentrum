"""
Quick demo of the pipeline visualization feature.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from augmentrum import Augmentrum

# Create some dummy data
print("Creating dummy MRS data...")
data_list = []
for i in range(5):
    data = np.random.randn(1, 1, 1, 2048, 8, 16) + 1j * np.random.randn(1, 1, 1, 2048, 8, 16)
    nifti = gen_nifti_mrs(data, 1/2000, 123.0)
    nifti.set_dim_tag(4, 'DIM_COIL')
    nifti.set_dim_tag(5, 'DIM_DYN')
    data_list.append(nifti)

print("✓ Created 5 dummy subjects\n")

# Example 1: Default pipeline
print("=" * 80)
print("EXAMPLE 1: Default Pipeline")
print("=" * 80)
augmenter1 = Augmentrum(data=data_list, batch_size=4)
augmenter1.show_pipeline(detailed=True)
print()

# Example 2: Custom pipeline with many modules
print("=" * 80)
print("EXAMPLE 2: Full Custom Pipeline")
print("=" * 80)
augmenter2 = Augmentrum(
    data=data_list,
    water=data_list,
    pipeline=[
        'processing',
        'phase',
        'frequency_shift',
        'line_broadening',
        'baseline',
        'residual_water',
        'eddy_current',
        'spurious_echoes',
        'noise',
        'apodization'
    ],
    backend='pytorch',
    batch_size=8,
    zero_order_deg=45.0,
    first_order_deg=90.0,
    shift_hz=5.0,
    lb_hz=4.0,
    gb_hz=2.5,
    baseline_frac=0.08,
    water_amp=0.15,
    eddy_mode='synthetic',
    sigma_frac=0.025,
    apod_mode='exponential'
)
augmenter2.show_pipeline(detailed=True)
print()

# Example 3: Minimal pipeline
print("=" * 80)
print("EXAMPLE 3: Minimal Pipeline (names only)")
print("=" * 80)
augmenter3 = Augmentrum(
    data=data_list,
    pipeline=['noise', 'line_broadening'],
    backend='numpy',
    batch_size=16
)
augmenter3.show_pipeline(detailed=False)
print()

print("✨ Demo complete! ✨")
