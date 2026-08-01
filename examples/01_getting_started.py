"""
================================================================================
                    AUGMENTRUM TUTORIAL 01: THE BASICS
================================================================================

📖 NEW TO AUGMENTRUM? Read 00_architecture_overview.py FIRST!
   It explains the complete data flow and answers key architectural questions.

WHAT IS AUGMENTRUM?
===================
Augmentrum is a framework for MRS (Magnetic Resonance Spectroscopy) data
augmentation. Think of it as a toolkit that lets you:

1. Load your MRS spectra (FID data from scanners)
2. Apply realistic transformations (noise, phase shifts, artifacts, etc.)
3. Generate infinite variations for training machine learning models
4. Export augmented data in standard formats

WHY USE AUGMENTRUM?
===================
Machine learning models need LOTS of data to train well. But MRS datasets are
usually small (tens or hundreds of subjects). Augmentrum solves this by:

• Creating realistic variations of your existing data
• Simulating common artifacts and imperfections
• Providing infinite training examples without collecting more data
• Making your ML models more robust to real-world variations

HOW DOES IT WORK?
=================
The framework has 4 main components working together:

┌─────────────────────────────────────────────────────────────────┐
│ 1. NIfTI-MRS+: Smart container for your spectra                 │
│    • Wraps FID data with metadata                               │
│    • Tracks what processing was done (provenance)               │
│    • Works with different data formats (numpy, pytorch, etc.)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Modules: Individual processing/augmentation operations       │
│    • PhaseShift: Rotate spectrum phase                          │
│    • GaussianNoise: Add realistic noise                         │
│    • LineBroadening: Broaden spectral lines                     │
│    • And 10+ more built-in modules                              │
│    • You can create your own custom modules!                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Pipelines: Chain modules together                            │
│    Example: processing → phase → noise → broadening             │
│    Each module processes the output of the previous one         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Augmentrum Class: Orchestrates everything                    │
│    • Loads your data                                            │
│    • Runs the pipeline                                          │
│    • Generates batches for training                             │
│    • Handles train/val/test splits                              │
└─────────────────────────────────────────────────────────────────┘

IN THIS TUTORIAL:
=================
We'll walk through the basics step by step:

Part 1: Loading real MRS data and understanding the data structure
Part 2: Building your first augmentation pipeline
Part 3: Understanding parameters (fixed values vs. random ranges)
Part 4: Visualizing augmentation effects with plots
Part 5: Creating your own custom augmentation module
Part 6: Putting it all together for training

Let's get started!
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add parent directory to path to import augmentrum
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from augmentrum import Augmentrum, AugmentationPipeline, BaseModule, Backend, NIfTI_MRS_Plus
from augmentrum.dataset.cows import COWSDataModule
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.augmentation.phase_frequency import PhaseShift, FrequencyShift
from augmentrum.augmentation.line_broadening import LineBroadening
from augmentrum.augmentation.gaussian_noise import GaussianNoise

# Import plotting utilities (now integrated into augmentrum)
import matplotlib.pyplot as plt
import numpy as np


# Simple plotting helper using native NIfTI-MRS plotting
def plot_comparison(nifti_list, labels, title, ppmlim=(0.2, 4.2)):
    """
    Plot multiple spectra overlaid in one figure with custom colors.
    Uses native NIfTI-MRS processing for correct handling.
    """
    import matplotlib.pyplot as plt
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.core.mrs import MRS

    # Define custom colors (avoid confusing default green)
    custom_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']  # blue, orange, red, purple, brown

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Convert each NIfTI-MRS to MRS object and plot
    for idx, (nifti, label) in enumerate(zip(nifti_list, labels)):
        # Process the data properly
        processed = nifti
        if 'DIM_COIL' in nifti.dim_tags and nifti.shape[nifti.dim_position('DIM_COIL')] > 1:
            processed = nifti_mrs_proc.coilcombine(nifti)

        # Average any remaining dimensions
        for dim in processed.dim_tags:
            if dim is not None and processed.shape[processed.dim_position(dim)] > 1:
                processed = nifti_mrs_proc.average(processed, dim)

        # Create MRS object
        mrs = MRS(
            processed[:].squeeze(),
            bw=processed.bandwidth,
            cf=processed.spectrometer_frequency[0],
            nucleus=processed.nucleus[0]
        )

        # Get spectrum data
        ppm = mrs.getAxes(ppmlim=ppmlim)
        spec = mrs.get_spec(ppmlim=ppmlim)

        # Plot with custom color
        color = custom_colors[idx % len(custom_colors)]
        ax.plot(ppm, spec.real, label=label, color=color, linewidth=1.5, alpha=0.8)

    # Format plot
    ax.invert_xaxis()
    ax.set_xlim(ppmlim[1], ppmlim[0])
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Amplitude (a.u.)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()
    return fig


#***********************************#
#   part 1: loading real mrs data   #
#***********************************#

print("\n" + "="*80)
print(" PART 1: LOADING REAL MRS DATA")
print("="*80)
print("""
Before we can augment data, we need to load it. Augmentrum works with real
scanner data in various formats (TWIX, SDAT, NIfTI-MRS, etc.).

For this tutorial, we'll use the COWS dataset - a publicly available brain
MRS dataset with healthy volunteers.

DATA STRUCTURE:
---------------
MRS data typically consists of:
• FID (Free Induction Decay): Time-domain signal from the scanner
• Metadata: Scanner parameters, sequence info, nucleus type, etc.
• Water reference: Separate acquisition for frequency/phase correction

The data might be:
• Multi-coil: Signal from each receiver coil separately
• Multi-average: Multiple acquisitions averaged for SNR
• Multi-voxel: If MRSI (spectroscopic imaging)
""")

# Configure data location
data_dir = os.environ.get(
    "DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openneuro_ds006812"))
)

print(f"Loading COWS dataset from: {data_dir}")
print("(Set DATA_DIR environment variable to use a different location)\n")

def load_cows_data(data_dir, location='PARIETAL', water_sup='VAPOR'):
    """Load COWS dataset using built-in data loader."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please set DATA_DIR environment variable or update data_dir in this script."
        )

    loader = COWSDataModule(data_dir=data_dir, location=location, water_sup=water_sup)
    data_list, water_list, _, _, names = loader.load_twix()

    if len(data_list) == 0:
        raise RuntimeError("No spectra loaded. Check data directory and parameters.")

    return data_list, water_list, names

# Load the data
data_list, water_list, subject_names = load_cows_data(data_dir)

print(f"✓ Successfully loaded {len(data_list)} subjects")
print(f"  Subject IDs: {subject_names[:5]}...")
print(f"\nDATA INFO:")
print(f"  • Each subject has metabolite + water reference acquisitions")
print(f"  • Format: NIfTI_MRS objects (FSL-MRS compatible)")
print(f"  • Data shape: {data_list[0].shape}")
print(f"  • Spectral points: {data_list[0].shape[-1]}")

# Visualize one raw spectrum
print("\nVisualizing a raw (unprocessed) spectrum...")
# Use native NIfTI-MRS plotting
data_list[0].plot(ppmlim=(0.2, 4.2), legend=True)
plt.gcf().suptitle("Raw MRS Spectrum - Before Any Processing", fontsize=12, fontweight='bold')
plt.show()


#*******************************************************#
#   part 2: building your first augmentation pipeline   #
#*******************************************************#

print("\n" + "="*80)
print(" PART 2: BUILDING YOUR FIRST AUGMENTATION PIPELINE")
print("="*80)
print("""
WHAT IS A PIPELINE?
-------------------
A pipeline is a sequence of operations (modules) applied to your data.
Think of it like a factory assembly line:

  Raw Data → Module 1 → Module 2 → Module 3 → Augmented Data

Each module does one thing (add noise, shift phase, etc.) and passes the
result to the next module.

COMMON PIPELINES:
-----------------
For training ML models, a typical pipeline might be:

1. 'processing' - Process raw scanner data (coil combination, etc.)
2. 'phase' - Random phase shifts (common scanner variation)
3. 'line_broadening' - Random line broadening (T2 variation)
4. 'noise' - Add realistic noise (SNR variation)

Let's build one!
""")

print("Creating augmentation pipeline...")

# Method 1: Using module names (easiest!)
augmenter_simple = Augmentrum(
    data=data_list[:3],  # Use just 3 subjects for this demo
    water=water_list[:3] if water_list else None,
    pipeline=['processing'],  # Just process raw data, no augmentation yet

    # Processing parameters for COWS data
    coil_method='fsl-mrs',   # Use FSL-MRS coil combination
    conj=False,              # Don't conjugate FIDs (COWS-specific)

    backend='nifti_list',
    batch_size=1,
    mode='fixed'
)

print("\n✓ Created simple pipeline with 1 module")
print("\nPIPELINE DETAILS:")
augmenter_simple.show_pipeline(detailed=True)

print("\n\nAVAILABLE MODULES:")
print("-" * 80)
print("Augmentrum has 15+ built-in modules you can use:")
print("")
modules = Augmentrum.AVAILABLE_MODULES
print("  PROCESSING:")
print("    • 'processing' or 'coil_sampling' - Process raw scanner data")
print("")
print("  SPECTRAL MANIPULATIONS:")
print("    • 'phase' - Zero/first-order phase shifts")
print("    • 'frequency_shift' - Frequency offset")
print("    • 'line_broadening' - Lorentzian/Gaussian broadening")
print("")
print("  NOISE:")
print("    • 'noise' or 'gaussian_noise' - Add white Gaussian noise")
print("")
print("  ARTIFACTS:")
print("    • 'baseline' - Baseline distortions")
print("    • 'residual_water' - Water suppression artifacts")
print("    • 'spurious_echoes' - Echo artifacts")
print("    • 'eddy_current' - Eddy current effects")
print("    • 'artificial_peaks' - Spurious peaks")
print("    • 'apodization' - Signal windowing")
print("")
print("  Just use the name in your pipeline list!")

# Generate a processed spectrum
print("\nGenerating processed spectrum (no augmentation yet)...")
batch_processed, _ = next(augmenter_simple.dataloader())
processed_nifti = batch_processed[0]

plot_comparison(
    [data_list[0], processed_nifti],
    ['Raw (Quick Processing for Visualization)', 'Processed'],
    "Effect of Processing Module"
)


#*********************************************************#
#   part 3: understanding parameters - fixed vs. random   #
#*********************************************************#

print("\n" + "="*80)
print(" PART 3: UNDERSTANDING PARAMETERS - THE KEY TO AUGMENTATION")
print("="*80)
print("""
This is THE MOST IMPORTANT concept in Augmentrum!

PARAMETER FORMATS:
------------------
Every module has parameters. You can specify them in two ways:

1. FIXED VALUE (scalar number):
   sigma_frac=0.02          → Always exactly 2% noise
   zero_order_deg=30.0      → Always exactly 30° phase shift
   
   Use this when you want reproducible, identical augmentations.

2. RANDOM RANGE (tuple of two numbers):
   sigma_frac=(0.01, 0.03)  → Random noise between 1-3% EACH TIME
   zero_order_deg=(0, 45)   → Random phase between 0-45° EACH TIME
   
   Use this for training - creates infinite variations!

WHY THIS MATTERS:
-----------------
When training ML models, you want to see DIFFERENT augmentations every epoch.
Random ranges make this happen automatically!

Example:
  • Epoch 1: spectrum gets 2.3% noise, 15° phase shift
  • Epoch 2: SAME spectrum gets 1.8% noise, 38° phase shift
  • Epoch 3: SAME spectrum gets 2.9% noise, 7° phase shift
  ... and so on forever!

This prevents overfitting and makes your model robust.

Let's see this in action with phase shifts:
""")

# Create a clean processed spectrum for comparison
clean_pipeline = AugmentationPipeline([NIfTI_RawProcessor(conj=False, coil_method='fsl-mrs')])
clean_data = NIfTI_MRS_Plus([data_list[0]], backend=Backend.NIFTI_LIST, volatile=True)
clean_water = NIfTI_MRS_Plus([water_list[0]], backend=Backend.NIFTI_LIST, volatile=True) if water_list else None
clean_data, _ = clean_pipeline(clean_data, clean_water)
clean_spectrum = clean_data.list()[0]

print("\nEXAMPLE 1: FIXED PHASE SHIFT (always 90 degrees)")
print("-" * 60)

# Fixed phase augmentation
aug_fixed = Augmentrum(
    data=[data_list[0]],  # Use SAME subject to show identical results
    water=[water_list[0]] if water_list else None,
    pipeline=['processing', 'phase'],
    zero_order_deg=90.0,  # FIXED: always 90 degrees

    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,

    mode='fixed',
    batch_size=1,
    backend='nifti_list'
)

# Generate multiple batches - they should be IDENTICAL
batch1, _ = next(aug_fixed.dataloader())
batch2, _ = next(aug_fixed.dataloader())

print("Generated 2 batches with FIXED parameters:")
print("  Both should look identical (same subject, same 90° phase shift)")

plot_comparison(
    [clean_spectrum, batch1[0], batch2[0]],
    ['Clean (0°)', 'Fixed 90° - Call 1', 'Fixed 90° - Call 2'],
    "Fixed Phase Shift: Always the Same!"
)

print("\n\nEXAMPLE 2: RANDOM PHASE SHIFT (0-90 degrees)")
print("-" * 60)

# Random phase augmentation
aug_random = Augmentrum(
    data=[data_list[0]],
    water=[water_list[0]] if water_list else None,
    pipeline=['processing', 'phase'],
    zero_order_deg=(0.0, 90.0),  # RANDOM: between 0-90 degrees

    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,

    mode='on-the-fly',  # Important for randomization!
    batch_size=1,
    backend='nifti_list'
)

batch1, _ = next(aug_random.dataloader())
batch2, _ = next(aug_random.dataloader())
batch3, _ = next(aug_random.dataloader())

print("Generated 3 batches with RANDOM parameters:")
print("  Each should look different (random phase 0-90°)")

plot_comparison(
    [clean_spectrum, batch1[0], batch2[0], batch3[0]],
    ['Clean', 'Random sample 1', 'Random sample 2', 'Random sample 3'],
    "Random Phase Shift: Different Every Time!"
)

print("\n\nKEY TAKEAWAY:")
print("  • Use scalars (e.g., 30.0) for fixed, reproducible augmentations")
print("  • Use tuples (e.g., (0, 45)) for random, varied augmentations")
print("  • Random ranges + mode='on-the-fly' = infinite training data!")


#**********************************************#
#   part 4: combining multiple augmentations   #
#**********************************************#

print("\n" + "="*80)
print(" PART 4: COMBINING MULTIPLE AUGMENTATIONS")
print("="*80)
print("""
Real-world MRS data varies in MANY ways:
• Phase drifts during acquisition
• Line widths vary with shimming quality
• Noise levels vary with coil coupling
• Frequency drifts from scanner instability

To make robust ML models, we simulate ALL these variations!

REALISTIC AUGMENTATION PIPELINE:
--------------------------------
processing → phase → broadening → noise

Let's see how each module affects the spectrum:
""")

subject = data_list[0]
subject_water = water_list[0] if water_list else None

# Clean
aug_clean = Augmentrum(
    data=[subject],
    water=[subject_water] if subject_water else None,
    pipeline=['processing'],
    coil_method='fsl-mrs',
    conj=False,
    mode='fixed',
    batch_size=1,
    backend='nifti_list'
)
clean, _ = next(aug_clean.dataloader())

# + Phase
aug_phase = Augmentrum(
    data=[subject],
    water=[subject_water] if subject_water else None,
    pipeline=['processing', 'phase'],
    zero_order_deg=25.0,
    coil_method='fsl-mrs',
    conj=False,
    mode='fixed',
    batch_size=1,
    backend='nifti_list'
)
with_phase, _ = next(aug_phase.dataloader())

# + Phase + Broadening
aug_broad = Augmentrum(
    data=[subject],
    water=[subject_water] if subject_water else None,
    pipeline=['processing', 'phase', 'line_broadening'],
    zero_order_deg=25.0,
    lb_hz=2.0,
    coil_method='fsl-mrs',
    conj=False,
    mode='fixed',
    batch_size=1,
    backend='nifti_list'
)
with_broad, _ = next(aug_broad.dataloader())

# + Phase + Broadening + Noise
aug_all = Augmentrum(
    data=[subject],
    water=[subject_water] if subject_water else None,
    pipeline=['processing', 'phase', 'line_broadening', 'noise'],
    zero_order_deg=25.0,
    lb_hz=2.0,
    sigma_frac=0.02,
    coil_method='fsl-mrs',
    conj=False,
    mode='fixed',
    batch_size=1,
    backend='nifti_list'
)
with_all, _ = next(aug_all.dataloader())

print("Visualizing cumulative effect of augmentations...")
plot_comparison(
    [clean[0], with_phase[0], with_broad[0], with_all[0]],
    ['Clean', '+ Phase (25°)', '+ Broadening (2Hz)', '+ Noise (2%)'],
    "Building Up Augmentations Step by Step"
)

print("\n\nNow with RANDOM parameters (what you'd use for training):")

aug_realistic = Augmentrum(
    data=data_list[:5],
    water=water_list[:5] if water_list else None,
    pipeline=['processing', 'phase', 'line_broadening', 'noise'],
    zero_order_deg=(0, 45),      # Random phase 0-45°
    lb_hz=(0.5, 3.0),            # Random broadening 0.5-3 Hz
    sigma_frac=(0.01, 0.03),     # Random noise 1-3%
    coil_method='fsl-mrs',
    conj=False,
    mode='on-the-fly',
    batch_size=4,
    backend='nifti_list'
)

print("\nGenerating 4 different augmented versions of the same spectrum:")
batch, _ = next(aug_realistic.dataloader())

plot_comparison(
    [clean[0], batch[0], batch[1], batch[2], batch[3]],
    ['Clean', 'Aug 1', 'Aug 2', 'Aug 3', 'Aug 4'],
    "Random Augmentations: Infinite Variety!"
)


#*********************************************#
#   part 5: creating your own custom module   #
#*********************************************#

print("\n" + "="*80)
print(" PART 5: CREATING YOUR OWN CUSTOM AUGMENTATION MODULE")
print("="*80)
print("""
The built-in modules cover most common augmentations, but you might want to
add your own! Maybe you want to simulate a specific artifact from your scanner,
or implement a novel augmentation strategy.

Creating a custom module is easy! You just need to:

1. Subclass BaseModule
2. Implement process_nifti_list() method (or process_tensor() or forward())
3. Use it like any built-in module

Let's create a simple "amplitude scaling" module that multiplies the FID by
a random gain factor. This simulates variations in receiver gain or coil
coupling.

CODE STRUCTURE:
---------------
class MyCustomModule(BaseModule):
    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]  # Which data formats it supports
    
    def __init__(self, my_param=1.0):
        super().__init__(my_param=my_param)
        self.my_param = my_param
    
    def process_nifti_list(self, data_list, water_list=None, **kwargs):
        # Your augmentation logic here
        for nifti in data_list:
            fid = nifti[:]  # Get FID data
            # Modify it
            nifti[:] = modified_fid  # Update
        return data_list, water_list
""")


#**************************************************************************************************#
#                                      Class AmplitudeScaling                                      #
#**************************************************************************************************#
#                                                                                                  #
# Custom module: Scale FID amplitude by a random factor.                                           #
#                                                                                                  #
#**************************************************************************************************#
class AmplitudeScaling(BaseModule):
    """
    Custom module: Scale FID amplitude by a random factor.

    Simulates variations in:
    - Receiver gain settings
    - Coil coupling
    - Sample loading

    Parameters:
    -----------
    scale_factor : float or tuple
        If float: fixed scaling factor
        If tuple: random scaling between (min, max)
    """
    SUPPORTED_BACKENDS = [Backend.NIFTI_LIST]

    def __init__(self, scale_factor=1.0):
        super().__init__(scale_factor=scale_factor)
        self.scale_factor = scale_factor

    def process_nifti_list(self, data_list, water_list=None, **kwargs):
        """Apply amplitude scaling to each spectrum."""
        import random

        processed = []
        for nifti in data_list:
            fid = nifti[:]

            # Sample scale factor if it's a range
            if isinstance(self.scale_factor, tuple):
                scale = random.uniform(*self.scale_factor)
            else:
                scale = self.scale_factor

            # Apply scaling
            nifti[:] = fid * scale
            processed.append(nifti)

        return processed, water_list

print("\n✓ Created custom AmplitudeScaling module!")
print("\nNow let's use it in a pipeline:")

# Use custom module with built-in modules
custom_pipeline = AugmentationPipeline([
    NIfTI_RawProcessor(coil_method='fsl-mrs', conj=False),  # Built-in with COWS params
    AmplitudeScaling(scale_factor=(0.8, 1.2)),  # YOUR custom module!
    PhaseShift(zero_order_deg=15.0),  # Built-in (using fixed value for demo)
    GaussianNoise(sigma_frac=0.02),   # Built-in
])

aug_custom = Augmentrum(
    data=[data_list[0]],
    water=[water_list[0]] if water_list else None,
    pipeline=custom_pipeline,  # Pass pipeline object directly
    mode='on-the-fly',
    batch_size=3,
    backend='nifti_list'
)

print("\nPipeline with custom module:")
aug_custom.show_pipeline(detailed=True)

batch_custom, _ = next(aug_custom.dataloader())

print("\nGenerating augmented spectra with custom module...")
plot_comparison(
    [clean[0], batch_custom[0], batch_custom[1], batch_custom[2]],
    ['Clean', 'Custom Aug 1', 'Custom Aug 2', 'Custom Aug 3'],
    "Using Your Custom Module in the Pipeline!"
)

print("\n\nCUSTOM MODULE TIPS:")
print("  • Implement process_nifti_list() for NIfTI data")
print("  • Implement process_tensor() for numpy/pytorch operations")
print("  • Implement forward() for maximum flexibility")
print("  • Support parameter ranges with isinstance(param, tuple) check")
print("  • Mix custom modules with built-ins freely!")


#****************************************************************#
#   part 6: putting it all together - realistic training setup   #
#****************************************************************#

print("\n" + "="*80)
print(" PART 6: PUTTING IT ALL TOGETHER FOR TRAINING")
print("="*80)
print("""
Now you know the basics! Let's set up a realistic augmentation pipeline for
training a machine learning model.

BEST PRACTICES FOR TRAINING:
-----------------------------
1. Use multiple augmentations (phase, noise, broadening, etc.)
2. Use random parameter ranges, not fixed values
3. Use mode='on-the-fly' for infinite variety
4. Use volatile=True to save memory (disables logging)
5. Adjust batch_size based on your GPU memory

Here's a complete example:
""")

# Realistic training augmenter
aug_training = Augmentrum(
    data=data_list,
    water=water_list,

    # Pipeline with multiple realistic augmentations
    pipeline=[
        'processing',        # Always start with processing!
        'phase',             # Phase variations
        'frequency_shift',   # Frequency drifts
        'line_broadening',   # T2 variations
        'noise'              # SNR variations
    ],

    # Random parameter ranges (will be sampled randomly each batch)
    zero_order_deg=(-45, 45),     # Phase: ±45 degrees
    first_order_deg=(-20, 20),    # First-order phase
    shift_hz=(-10, 10),           # Frequency: ±10 Hz
    lb_hz=(0.5, 3.0),             # Line broadening: 0.5-3 Hz
    gb_hz=(0, 1.0),               # Gaussian broadening: 0-1 Hz
    sigma_frac=(0.005, 0.03),     # Noise: 0.5-3%

    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,

    # Training settings
    mode='on-the-fly',     # Random augmentations each batch
    batch_size=8,          # Adjust for your GPU
    backend='numpy',       # Or 'pytorch', 'tensorflow', etc.
    volatile=True          # Save memory, don't log provenance
)

print("\n✓ Created realistic training augmenter")
print("\nCONFIGURATION:")
print(f"  • Dataset size: {len(data_list)} subjects")
print(f"  • Batch size: {aug_training.batch_size}")
print(f"  • Mode (train): {aug_training.modes['train']}")
print(f"  • Backend: {aug_training.backend.value}")
print("\nPIPELINE:")
aug_training.show_pipeline(detailed=True)

print("\n\nGenerating training batches...")
print("(Each batch will have DIFFERENT augmentations)")

# Generate a few batches
dataloader = aug_training.dataloader(framework='numpy')

for i in range(3):
    batch_data, batch_water = next(dataloader)
    print(f"\nBatch {i+1}:")
    print(f"  Shape: {batch_data.shape}")
    print(f"  Type: {type(batch_data)}")
    print(f"  Min/Max: [{batch_data.min():.2e}, {batch_data.max():.2e}]")

print("\n\nUSAGE IN TRAINING LOOP:")
print("""
# Pseudocode for training:
for epoch in range(num_epochs):
    for batch_data, batch_water in aug_training.dataloader():
        # batch_data shape: (batch_size, n_points)
        # Each batch has different augmentations!
        
        predictions = model(batch_data)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
""")


#*********************************************************#
#   part 7: pre-pipeline - caching expensive operations   #
#*********************************************************#

print("\n" + "="*80)
print(" PART 7: PRE-PIPELINE - EFFICIENT PREPROCESSING")
print("="*80)
print("""
FEATURE: Pre-Pipeline! 🚀

PROBLEM:
--------
Some preprocessing steps are EXPENSIVE (like coil combination, eddy current
correction) but don't need to be randomized. Running them every batch wastes time!

SOLUTION: PRE-PIPELINE
-----------------------
Pre-pipeline runs ONCE during initialization and caches the results.
Then on-the-fly augmentation runs on the preprocessed data.

WORKFLOW:
---------
1. Load raw data (multi-coil, multi-average)
2. Pre-pipeline: Expensive operations (run ONCE, cached)
   • Coil combination
   • Average selection
   • Eddy current correction
3. Main pipeline: Fast augmentations (run on-the-fly)
   • Phase shifts
   • Line broadening
   • Noise addition

BENEFITS:
---------
✓ Faster training (expensive ops run once, not every batch!)
✓ Still get infinite variety from on-the-fly augmentation
✓ Best of both worlds!

Let's see it in action:
""")

print("\nEXAMPLE: Without Pre-Pipeline (slower)")
print("-" * 60)
print("Every batch runs: coil sampling + processing + phase + noise")
print("This is SLOW because coil/average sampling happens every time!")

# This would be inefficient (but works)
aug_slow = Augmentrum(
    data=data_list[:3],
    water=water_list[:3] if water_list else None,

    pipeline=['coil_sampling', 'processing', 'phase', 'noise'],

    # Parameters for main pipeline (sampled every batch)
    n_coils=(16, 32),           # Random coil selection EVERY batch
    n_averages=(16, 64),       # Random average selection EVERY batch
    zero_order_deg=(0, 45),    # Random phase EVERY batch
    sigma_frac=(0.01, 0.03),   # Random noise EVERY batch

    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,

    mode='on-the-fly',
    batch_size=2,
    backend='numpy',
    volatile=True
)

print("\n✓ Created (but this is inefficient for training!)")

print("\n\nEXAMPLE: With Pre-Pipeline (faster!)")
print("-" * 60)
print("Pre-pipeline (run ONCE, cached):")
print("  • coil_sampling - select and combine coils")
print("  • processing - process raw FID data")
print("")
print("Main pipeline (run on-the-fly):")
print("  • phase - random phase shifts")
print("  • noise - random noise levels")

# This is MUCH more efficient!
aug_fast = Augmentrum(
    data=data_list[:3],
    water=water_list[:3] if water_list else None,

    # Pre-pipeline: Run ONCE and cache
    pre_pipeline=['coil_sampling', 'processing'],

    # Main pipeline: Run on-the-fly (fast operations only!)
    pipeline=['phase', 'noise'],

    # Parameters for pre-pipeline (sampled once during init)
    n_coils=(16, 32),  # Random coil selection ONCE during initialization
    n_averages=(16, 64),  # Random average selection ONCE during initialization

    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,

    # Parameters for main pipeline (sampled every batch)
    zero_order_deg=(0, 45),    # Random phase EVERY batch
    sigma_frac=(0.01, 0.03),   # Random noise EVERY batch

    mode='on-the-fly',
    batch_size=2,
    backend='numpy',
    volatile=True
)

print("\n✓ Pre-pipeline applied and cached!")
print("\nNow generating batches (fast!)...")

# Generate batches - notice how fast this is!
batch1, _ = next(aug_fast.dataloader())
batch2, _ = next(aug_fast.dataloader())

print(f"Generated 2 batches with different augmentations")
print(f"But the expensive preprocessing was done ONCE!")

# Visualize the difference
plot_comparison(
    [batch1[0], batch2[0]],
    ['Batch 1 (random aug)', 'Batch 2 (random aug)'],
    "Pre-Pipeline: Preprocessed Once, Augmented On-the-Fly"
)

print("\n\nPERFORMANCE COMPARISON:")
print("-" * 60)
print("""
WITHOUT PRE-PIPELINE:
  Every batch: coil_sampling (slow) + processing (slow) + phase + noise
  Total time per batch: ~10-50 ms

WITH PRE-PIPELINE:
  Initialization: coil_sampling + processing (run once for all data)
  Every batch: phase + noise (fast operations only!)
  Total time per batch: ~1-5 ms
  
SPEEDUP: 5-10x faster during training! 🚀
""")

print("\n\nWHEN TO USE PRE-PIPELINE:")
print("-" * 60)
print("""
USE PRE-PIPELINE FOR:
✓ Coil combination (expensive)
✓ Average selection/combination (expensive)
✓ Eddy current correction (expensive)
✓ Any operation that doesn't need randomization
✓ Operations where you want fixed preprocessing

USE MAIN PIPELINE FOR:
✓ Phase shifts (fast, should be random)
✓ Frequency shifts (fast, should be random)
✓ Line broadening (fast, should be random)
✓ Noise addition (fast, should be random)
✓ Any augmentation that creates variety
""")

print("\n\nRECOMMENDED WORKFLOW:")
print("-" * 60)
print("""
For training on raw multi-coil data:

aug = Augmentrum(
    data=raw_data_list,
    water=raw_water_list,
    
    # Pre-pipeline: Fixed, expensive operations (cached)
    pre_pipeline=['coil_sampling', 'processing'],
    n_coils=8,           # or (4, 16) to sample once
    n_averages=32,       # or (16, 64) to sample once
    
    # Processing parameters (for COWS data)
    coil_method='fsl-mrs',
    conj=False,
    
    # Main pipeline: Fast, random augmentations  
    pipeline=['phase', 'frequency_shift', 'line_broadening', 'noise'],
    zero_order_deg=(-45, 45),     # Random every batch
    shift_hz=(-10, 10),           # Random every batch
    lb_hz=(0.5, 3.0),             # Random every batch
    sigma_frac=(0.01, 0.03),      # Random every batch
    
    mode='on-the-fly',
    backend='pytorch',
    volatile=True
)

Result: Fast training with infinite variety! 🎉
""")


#****************************#
#   summary and next steps   #
#****************************#

print("\n" + "="*80)
print(" 🎉 TUTORIAL COMPLETE!")
print("="*80)
print("""
CONGRATULATIONS! You now understand:

✓ What Augmentrum is and why it's useful
✓ How to load MRS data
✓ How to build augmentation pipelines
✓ The difference between fixed and random parameters
✓ How to combine multiple augmentations
✓ How to create custom modules
✓ How to set up augmentation for training
✓ How to use pre-pipeline for efficient preprocessing ← NEW!

KEY CONCEPTS TO REMEMBER:
-------------------------
1. NIfTI-MRS+: Smart container for spectra with metadata tracking
2. Modules: Individual operations (phase, noise, etc.)
3. Pipelines: Chain modules together (module1 → module2 → ...)
4. Parameters: Scalars = fixed, tuples = random ranges
5. Modes: 'on-the-fly' = random, 'fixed' = reproducible
6. Pre-Pipeline: Cache expensive operations, augment fast ones ← NEW!

PARAMETER CHEAT SHEET:
----------------------
sigma_frac=0.02           → Always 2% noise
sigma_frac=(0.01, 0.03)   → Random 1-3% noise
mode='fixed'              → Same augmentation every batch
mode='on-the-fly'         → Different augmentation every batch

PRE-PIPELINE PATTERN:
---------------------
pre_pipeline=['coil_sampling', 'processing']  # Run once, cache
pipeline=['phase', 'noise', 'broadening']      # Run every batch

NEXT TUTORIALS:
---------------
• 02_splits_and_dataloaders.py - Learn about train/val/test splits
• 03_backends_and_training.py - Use with PyTorch, TensorFlow, etc.

READY TO USE AUGMENTRUM?
------------------------
Just copy this template and modify the parameters:

aug = Augmentrum(
    data=your_data_list,
    water=your_water_list,
    
    # Pre-pipeline for expensive ops (optional but recommended!)
    pre_pipeline=['coil_sampling', 'processing'],
    n_coils=8,
    n_averages=32,
    
    # Processing parameters (adjust for your data)
    coil_method='fsl-mrs',  # For COWS data
    conj=False,             # For COWS data
    
    # Main pipeline for fast augmentations
    pipeline=['phase', 'line_broadening', 'noise'],
    zero_order_deg=(0, 45),
    lb_hz=(0.5, 3.0),
    sigma_frac=(0.01, 0.03),
    
    mode='on-the-fly',
    batch_size=32,
    backend='pytorch',  # or 'numpy', 'tensorflow', etc.
    volatile=True
)

for batch, water in aug.dataloader():
    # Train your model!
    pass

Happy augmenting! 🚀
""")









