"""
================================================================================
            AUGMENTRUM TUTORIAL 02: DATA SPLITS & DATALOADERS
================================================================================

📖 PREREQUISITE: Complete Tutorial 01 (01_getting_started.py) first!
   For architecture details, see 00_architecture_overview.py

WHAT YOU'LL LEARN:
==================
In Tutorial 01, you learned the basics of Augmentrum. Now we'll cover the
essential concepts for training machine learning models:

• How to split your data into train/validation/test sets
• Why different splits need different augmentation strategies
• How to use dataloaders to generate batches
• Understanding 'on-the-fly' vs 'fixed' modes in depth
• What provenance logging is and when to disable it (volatile mode)
• How to save and load augmented data

WHY DATA SPLITS MATTER:
=======================
When training ML models, you need to evaluate them properly:

┌──────────────────────────────────────────────────────────────┐
│ TRAIN SET (70-80% of data)                                   │
│ • Used to train the model                                    │
│ • HEAVY augmentation (noise, phase, broadening, etc.)        │
│ • mode='on-the-fly' → different augmentations every epoch    │
│ • Goal: See maximum variety to prevent overfitting           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ VALIDATION SET (10-15% of data)                              │
│ • Used to tune hyperparameters and monitor training          │
│ • LIGHT or NO augmentation (just processing)                 │
│ • mode='fixed' → same augmentations every epoch              │
│ • Goal: Consistent evaluation to track true performance      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ TEST SET (10-15% of data)                                    │
│ • Used for final model evaluation                            │
│ • NO augmentation (raw or minimal processing)                │
│ • mode='fixed' → reproducible results                        │
│ • Goal: Unbiased estimate of real-world performance          │
└──────────────────────────────────────────────────────────────┘

HOW AUGMENTRUM HANDLES SPLITS:
===============================
Augmentrum makes this easy:

1. Define split fractions: split_fractions={'val': 0.15, 'test': 0.15}
2. Assign pipelines per split: pipelines={'train': [...], 'val': [...]}
3. Set modes per split: modes={'train': 'on-the-fly', 'val': 'fixed'}
4. Access split-specific dataloaders: train_dataloader(), val_dataloader()

Let's see this in action!
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from augmentrum import Augmentrum, Backend
from augmentrum.dataset.cows import COWSDataModule


#***************************#
#   visualization helpers   #
#***************************#
def plot_batch_variety(batch_data, title, max_plots=6):
    """
    Plot multiple spectra from a batch to show variability.
    Uses custom processing and grid layout like example 01.
    """
    import matplotlib.pyplot as plt
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.core.mrs import MRS
    
    # Handle different input types
    if hasattr(batch_data, 'list'):
        nifti_list = batch_data.list()
    elif isinstance(batch_data, list):
        nifti_list = batch_data
    else:
        raise TypeError(f"Expected NIfTI_MRS_Plus or list, got {type(batch_data)}")
    
    n_to_plot = min(len(nifti_list), max_plots)
    
    # Grid layout
    if n_to_plot <= 3:
        rows, cols = 1, n_to_plot
    elif n_to_plot <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    axes = axes.flatten()
    
    # Define custom colors
    custom_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i in range(n_to_plot):
        ax = axes[i]
        nifti = nifti_list[i]
        
        # Process the data
        processed = nifti
        if 'DIM_COIL' in nifti.dim_tags and nifti.shape[nifti.dim_position('DIM_COIL')] > 1:
            processed = nifti_mrs_proc.coilcombine(nifti)
        
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
        
        # Get spectrum
        ppm = mrs.getAxes(ppmlim=(0.5, 4.2))
        spec = mrs.get_spec(ppmlim=(0.5, 4.2))
        
        # Plot with custom color
        color = custom_colors[i % len(custom_colors)]
        ax.plot(ppm, spec.real, color=color, linewidth=1.2)
        
        # Format
        ax.invert_xaxis()
        ax.set_xlim(4.2, 0.5)
        ax.set_xlabel("ppm", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_title(f"Spectrum {i+1}", fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Hide unused subplots
    for i in range(n_to_plot, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig


def plot_overlay(nifti_list, labels, title):
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
        ppm = mrs.getAxes(ppmlim=(0.5, 4.2))
        spec = mrs.get_spec(ppmlim=(0.5, 4.2))

        # Plot with custom color
        color = custom_colors[idx % len(custom_colors)]
        ax.plot(ppm, spec.real, label=label, color=color, linewidth=1.5, alpha=0.8)

    # Format plot
    ax.invert_xaxis()
    ax.set_xlim(4.2, 0.5)
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Amplitude (a.u.)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()
    return fig


#****************************************#
#   part 1: loading data for splitting   #
#****************************************#
print("\n" + "="*80)
print(" PART 1: LOADING DATA")
print("="*80)
print("""
First, let's load our dataset. For meaningful splits, we want enough subjects
in each split. The COWS dataset is perfect for this demonstration.
""")

data_dir = os.environ.get(
    "DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openneuro_ds006812"))
)

print(f"Loading COWS dataset from: {data_dir}\n")

def load_cows_data(data_dir, location='PARIETAL', water_sup='VAPOR'):
    """Load COWS dataset."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data not found: {data_dir}")
    loader = COWSDataModule(data_dir=data_dir, location=location, water_sup=water_sup)
    data_list, water_list, _, _, names = loader.load_twix()
    if len(data_list) == 0:
        raise RuntimeError("No data loaded.")
    return data_list, water_list, names

data_list, water_list, subject_names = load_cows_data(data_dir)

print(f"✓ Loaded {len(data_list)} subjects")
print(f"  This is enough for a 70/15/15 train/val/test split!")
print(f"  Subject IDs: {subject_names[:5]}...\n")


#*****************************#
#   part 2: creating splits   #
#*****************************#
print("="*80)
print(" PART 2: CREATING TRAIN/VAL/TEST SPLITS")
print("="*80)
print("""
HOW SPLITTING WORKS:
--------------------
Augmentrum randomly assigns subjects to train/val/test sets based on the
fractions you specify. The 'seed' parameter ensures reproducibility.

EXAMPLE:
  split_fractions={'val': 0.15, 'test': 0.15}
  
This means:
  • 15% of subjects → validation set
  • 15% of subjects → test set
  • Remaining 70% → training set

The split is done at the SUBJECT level, not spectrum level. This is crucial
because spectra from the same subject are correlated - you don't want data
from the same person in both train and test sets!

Let's create splits:
""")

augmenter_with_splits = Augmentrum(
    data=data_list,
    water=water_list,

    # Define splits (seed for reproducibility)
    split_fractions={'val': 0.2, 'test': 0.1},  # 70% train, 20% val, 10% test
    seed=42,

    # For now, just use simple processing (we'll add augmentations later)
    pipeline=['processing'],

    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,

    backend='nifti_list',
    batch_size=4,
    mode='fixed'
)

print("\n✓ Created splits!")
print("\nSPLIT DISTRIBUTION:")
print(f"  • Total subjects: {len(data_list)}")

# Access split data correctly
train_data, _ = augmenter_with_splits.splits['train']
val_data, _ = augmenter_with_splits.splits['val']
test_data, _ = augmenter_with_splits.splits['test']

print(f"  • Training:   {len(train_data)} subjects "
      f"({len(train_data)/len(data_list)*100:.1f}%)")
print(f"  • Validation: {len(val_data)} subjects "
      f"({len(val_data)/len(data_list)*100:.1f}%)")
print(f"  • Test:       {len(test_data)} subjects "
      f"({len(test_data)/len(data_list)*100:.1f}%)")

# Get subject names from the actual data
train_subjects = [nifti.image_metadata()['PatientID'] if hasattr(nifti, 'image_metadata')
                  else f"subject_{i}" for i, nifti in enumerate(train_data.list()[:3])]
val_subjects = [nifti.image_metadata()['PatientID'] if hasattr(nifti, 'image_metadata')
                else f"subject_{i}" for i, nifti in enumerate(val_data.list()[:3])]

print("\nTRAIN SUBJECTS (first 3):", train_subjects, "...")
print("VAL SUBJECTS (first 3):  ", val_subjects, "...")

# Get test subjects similarly
test_subjects = [nifti.image_metadata()['PatientID'] if hasattr(nifti, 'image_metadata')
                 else f"subject_{i}" for i, nifti in enumerate(test_data.list()[:2])]
print("TEST SUBJECTS:  ", test_subjects, "...")

print("""
KEY POINT: Same seed = same split every time!
This ensures you can reproduce your experiments exactly.
""")


#*******************************************#
#   part 3: different pipelines per split   #
#*******************************************#
print("\n" + "="*80)
print(" PART 3: DIFFERENT AUGMENTATION STRATEGIES PER SPLIT")
print("="*80)
print("""
WHY DIFFERENT PIPELINES?
------------------------
Training, validation, and test sets serve different purposes:

TRAINING:
  • Goal: Learn robust features
  • Strategy: HEAVY augmentation to see maximum variety
  • Pipeline: processing + phase + broadening + noise + artifacts

VALIDATION:
  • Goal: Monitor training progress consistently
  • Strategy: LIGHT/NO augmentation for consistent evaluation
  • Pipeline: processing only (or processing + minimal augmentation)

TEST:
  • Goal: Unbiased final evaluation
  • Strategy: NO augmentation (real-world conditions)
  • Pipeline: None or just processing

Let's set this up:
""")

augmenter_split_pipelines = Augmentrum(
    data=data_list,
    water=water_list,

    # Splits
    split_fractions={'val': 0.2, 'test': 0.1},
    seed=42,

    # DIFFERENT PIPELINE FOR EACH SPLIT!
    pipelines={
        'train': ['processing', 'phase', 'line_broadening', 'noise'],  # Heavy
        'val': ['processing', 'noise'],  # Minimal
        'test': ['processing']  # Only processing, no augmentation
    },

    # Augmentation parameters (only used by train pipeline)
    zero_order_deg=(0, 45),
    lb_hz=(0.5, 3.0),
    sigma_frac=(0.01, 0.03),

    backend='nifti_list',
    batch_size=4,
    mode='fixed'  # We'll discuss modes in next section
)

print("\n✓ Created split-specific pipelines!")
print("\nTRAIN PIPELINE (Heavy Augmentation):")
augmenter_split_pipelines.show_pipeline(split='train', detailed=False)

print("\nVAL PIPELINE (Minimal):")
augmenter_split_pipelines.show_pipeline(split='val', detailed=False)

print("\nTEST PIPELINE (None):")
if augmenter_split_pipelines.pipelines['test'] is None:
    print("  No pipeline (raw data)")
else:
    augmenter_split_pipelines.show_pipeline(split='test', detailed=False)

print("""
This setup ensures:
  ✓ Training sees maximum variety (prevents overfitting)
  ✓ Validation is consistent (reliable performance monitoring)
  ✓ Test is unbiased (true real-world performance)
""")


#*******************************************************#
#   part 4: understanding modes - on-the-fly vs fixed   #
#*******************************************************#
print("\n" + "="*80)
print(" PART 4: ON-THE-FLY VS FIXED MODE - THE CRITICAL DIFFERENCE")
print("="*80)
print("""
This is CRUCIAL for training! Let's understand the two modes:

MODE='ON-THE-FLY':
------------------
• Parameters with ranges (e.g., (0, 45)) are sampled FRESHLY every batch
• Same subject gets DIFFERENT augmentations each epoch
• Creates infinite variety from finite data
• Essential for training!

Example: Subject A in epoch 1 gets: 12° phase, 1.8 Hz broadening, 2.1% noise
         Subject A in epoch 2 gets: 38° phase, 2.7 Hz broadening, 1.3% noise
         Subject A in epoch 3 gets: 5° phase, 0.9 Hz broadening, 2.8% noise
         ... forever!

MODE='FIXED':
-------------
• Parameters with ranges are sampled ONCE and reused
• Same subject gets SAME augmentations each epoch
• Reproducible and consistent
• Used for validation/testing!

Example: Subject B always gets: 27° phase, 2.1 Hz broadening, 1.9% noise
         (same every time you call the dataloader)

Let's see the difference visually:
""")

print("\nDEMONSTRATION 1: ON-THE-FLY MODE")
print("-" * 60)

aug_otf = Augmentrum(
    data=[data_list[0]],  # Just one subject
    water=[water_list[0]] if water_list else None,  # Add water reference
    pipeline=['processing', 'phase', 'noise'],
    zero_order_deg=(0, 90),
    sigma_frac=(0.01, 0.03),
    coil_method='fsl-mrs',
    conj=False,
    mode='on-the-fly',  # ← THIS IS THE KEY
    batch_size=1,
    backend='nifti_list'
)

print("Generating 6 batches from the SAME subject with on-the-fly mode...")
print("Each should look DIFFERENT:")

otf_batches = []
for i in range(6):
    batch, _ = next(aug_otf.dataloader())
    otf_batches.append(batch[0])

plot_batch_variety(otf_batches,
                   "On-The-Fly Mode: Same Subject, Different Every Time!")

print("\n\nDEMONSTRATION 2: FIXED MODE")
print("-" * 60)

aug_fixed = Augmentrum(
    data=[data_list[0]],  # Same subject
    water=[water_list[0]] if water_list else None,  # Add water reference
    pipeline=['processing', 'phase', 'noise'],
    zero_order_deg=(0, 90),  # Same parameter ranges
    sigma_frac=(0.01, 0.03),
    coil_method='fsl-mrs',
    conj=False,
    mode='fixed',  # ← DIFFERENT MODE
    batch_size=1,
    backend='nifti_list'
)

print("Generating 6 batches from the SAME subject with fixed mode...")
print("Each should look SIMILAR (same augmentation parameters):")

fixed_batches = []
for i in range(6):
    batch, _ = next(aug_fixed.dataloader())
    fixed_batches.append(batch[0])

plot_batch_variety(fixed_batches,
                   "Fixed Mode: Same Subject, Same Parameters Every Time!")

print("\n\nDirect comparison of first batch from each mode:")
plot_overlay(
    [otf_batches[0], fixed_batches[0]],
    ['On-the-fly (random)', 'Fixed (same params)'],
    "On-The-Fly vs Fixed: Direct Comparison"
)

print("""
KEY TAKEAWAY:
  • Use mode='on-the-fly' for TRAINING (infinite variety)
  • Use mode='fixed' for VALIDATION/TESTING (reproducibility)
""")


#*******************************#
#   part 5: using dataloaders   #
#*******************************#
print("\n" + "="*80)
print(" PART 5: USING DATALOADERS FOR BATCH GENERATION")
print("="*80)
print("""
Dataloaders are Python generators that produce batches of augmented data.
They're designed to work seamlessly in training loops.

SPLIT-SPECIFIC DATALOADERS:
---------------------------
• train_dataloader() → batches from training set
• val_dataloader() → batches from validation set
• test_dataloader() → batches from test set

Each dataloader respects its split's pipeline and mode!

Let's create a realistic training setup:
""")

aug_training = Augmentrum(
    data=data_list,
    water=water_list,

    # Splits
    split_fractions={'val': 0.2, 'test': 0.1},
    seed=42,

    # Different pipelines
    pipelines={
        'train': ['processing', 'phase', 'line_broadening', 'noise'],
        'val': ['processing'],
        'test': None
    },

    # Different modes!
    modes={
        'train': 'on-the-fly',  # Different every epoch
        'val': 'fixed',         # Same every epoch
        'test': 'fixed'         # Same every epoch
    },

    # Augmentation parameters
    zero_order_deg=(-45, 45),
    lb_hz=(0.5, 3.0),
    sigma_frac=(0.005, 0.03),

    # Settings
    batch_size=8,
    backend='numpy',
    volatile=True  # We'll explain this next!
)

print("\n✓ Created training-ready augmenter")
print(f"\nCONFIGURATION:")
print(f"  Total: {len(data_list)} subjects")
print(f"  Train: {len(aug_training.splits['train'][0])} subjects, on-the-fly mode")
print(f"  Val:   {len(aug_training.splits['val'][0])} subjects, fixed mode")
print(f"  Test:  {len(aug_training.splits['test'][0])} subjects, fixed mode")

print("\n\nGENERATING BATCHES:")
print("-" * 60)

# Training batches
print("\nTRAINING BATCHES (on-the-fly mode):")
train_dl = aug_training.train_dataloader(framework='numpy')
for i in range(2):
    batch_data, batch_water = next(train_dl)
    print(f"  Batch {i+1}: shape={batch_data.shape}, "
          f"mean={batch_data.mean():.3e}, std={batch_data.std():.3e}")

# Validation batches
print("\nVALIDATION BATCHES (fixed mode):")
val_dl = aug_training.val_dataloader(framework='numpy')
for i in range(1):  # Only 1 batch since val set is small (2 subjects, batch_size=8)
    batch_data, batch_water = next(val_dl)
    print(f"  Batch {i+1}: shape={batch_data.shape}, "
          f"mean={batch_data.mean():.3e}, std={batch_data.std():.3e}")

print("""
TYPICAL TRAINING LOOP:
----------------------
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch_data, batch_water in aug_training.train_dataloader():
        predictions = model(batch_data)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch_data, _ in aug_training.val_dataloader():
            val_loss = model(batch_data)
""")


#*************************************************#
#   part 6: provenance logging vs volatile mode   #
#*************************************************#
print("\n" + "="*80)
print(" PART 6: PROVENANCE LOGGING VS VOLATILE MODE")
print("="*80)
print("""
WHAT IS PROVENANCE LOGGING?
----------------------------
By default, Augmentrum tracks EVERY operation applied to your data:
• Which modules were used
• What parameter values were applied
• Complete processing history

This metadata is stored in the NIfTI-MRS+ container and can be saved with
the data. It's useful for reproducibility and debugging!

EXAMPLE:
  Spectrum metadata shows:
    1. NIfTI_RawProcessor applied
    2. PhaseShift applied (zero_order=27.3°)
    3. GaussianNoise applied (sigma_frac=0.021)
    ... complete audit trail!

VOLATILE MODE (volatile=True):
-------------------------------
Disables provenance logging for SPEED and MEMORY efficiency.

Use volatile=True when:
  ✓ Training (you don't need to save every batch)
  ✓ On-the-fly generation (parameters change constantly anyway)
  ✓ Memory is limited

Use volatile=False when:
  ✓ Creating a dataset to share
  ✓ Debugging augmentations
  ✓ Need reproducibility/audit trail

Let's compare:
""")

print("\nEXAMPLE 1: WITH LOGGING (volatile=False)")
print("-" * 60)

aug_logged = Augmentrum(
    data=[data_list[0]],
    water=[water_list[0]] if water_list else None,  # Add water reference
    pipeline=['processing', 'phase', 'noise'],
    zero_order_deg=30.0,
    sigma_frac=0.02,
    coil_method='fsl-mrs',
    conj=False,
    backend='nifti_list',
    batch_size=1,
    mode='fixed',
    volatile=False  # ← Logging enabled
)

batch_logged, _ = next(aug_logged.dataloader())
nifti_logged = batch_logged[0]

print("✓ Generated batch WITH provenance logging")
print("  Use case: Creating a dataset to share or export")
print("  Metadata is tracked and can be saved with the data")

print("\n\nEXAMPLE 2: WITHOUT LOGGING (volatile=True)")
print("-" * 60)

aug_volatile = Augmentrum(
    data=[data_list[0]],
    water=[water_list[0]] if water_list else None,  # Add water reference
    pipeline=['processing', 'phase', 'noise'],
    zero_order_deg=30.0,
    sigma_frac=0.02,
    coil_method='fsl-mrs',
    conj=False,
    backend='nifti_list',
    batch_size=1,
    mode='fixed',
    volatile=True  # ← Logging disabled
)

batch_volatile, _ = next(aug_volatile.dataloader())
nifti_volatile = batch_volatile[0]

print("✓ Generated batch WITHOUT provenance logging")
print("  Use case: Training (faster, less memory)")
print("  No metadata tracking")

print("""
RECOMMENDATION:
  • Training: volatile=True (faster, less memory)
  • Validation: volatile=True (usually don't need to save)
  • Creating datasets: volatile=False (track what you did)
  • Debugging: volatile=False (see exactly what happened)
""")


#***************************************#
#   part 7: complete training example   #
#***************************************#
print("\n" + "="*80)
print(" PART 7: PUTTING IT ALL TOGETHER - COMPLETE TRAINING SETUP")
print("="*80)
print("""
Let's create a complete, production-ready augmentation setup for training
a neural network. This incorporates all best practices:
""")

augmenter_complete = Augmentrum(
    data=data_list,
    water=water_list,

    #****************************#
    #   1. split configuration   #
    #****************************#
    split_fractions={'val': 0.15, 'test': 0.15},  # 70/15/15 split
    seed=42,  # Reproducible splits

    #*********************************#
    #   2. split-specific pipelines   #
    #*********************************#
    pipelines={
        'train': [
            'processing',
            'phase',
            'frequency_shift',
            'line_broadening',
            'noise'
        ],
        'val': ['processing'],
        'test': None
    },

    #*****************************#
    #   3. split-specific modes   #
    #*****************************#
    modes={
        'train': 'on-the-fly',  # Infinite variety
        'val': 'fixed',         # Consistent evaluation
        'test': 'fixed'         # Reproducible results
    },

    #********************************#
    #   4. augmentation parameters   #
    #********************************#
    # These ranges are sampled randomly for training
    zero_order_deg=(-45, 45),      # Phase ±45°
    first_order_deg=(-20, 20),     # First-order phase
    shift_hz=(-10, 10),            # Frequency drift
    lb_hz=(0.5, 3.0),              # Lorentzian broadening
    gb_hz=(0, 1.0),                # Gaussian broadening
    sigma_frac=(0.005, 0.03),      # Noise 0.5-3%

    #*************************#
    #   5. general settings   #
    #*************************#
    batch_size=32,      # Adjust for your GPU
    backend='numpy',    # Or 'pytorch', 'tensorflow'
    volatile=True       # Fast, no logging
)

print("\n✓ Created production-ready augmenter")
print("\nCONFIGURATION SUMMARY:")
print(f"  Dataset: {len(data_list)} subjects")
print(f"  Train:   {len(augmenter_complete.splits['train'][0])} subjects (on-the-fly augmentation)")
print(f"  Val:     {len(augmenter_complete.splits['val'][0])} subjects (fixed, minimal aug)")
print(f"  Test:    {len(augmenter_complete.splits['test'][0])} subjects (fixed, no aug)")
print(f"  Batch size: {augmenter_complete.batch_size}")
print(f"  Backend: {augmenter_complete.backend.name}")

print("\n\nTRAINING PIPELINE:")
augmenter_complete.show_pipeline(split='train', detailed=True)

print("\n\nPSEUDOCODE FOR TRAINING:")
print("""
import torch
import torch.nn as nn

# Your model
model = YourNeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_data, batch_water in augmenter_complete.train_dataloader():
        # Convert to torch if needed
        batch_data = torch.from_numpy(batch_data).float()
        
        # Forward pass
        predictions = model(batch_data)
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_data, _ in augmenter_complete.val_dataloader():
            batch_data = torch.from_numpy(batch_data).float()
            predictions = model(batch_data)
            loss = criterion(predictions, targets)
            val_loss += loss.item()
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

# Final testing
model.eval()
with torch.no_grad():
    for batch_data, _ in augmenter_complete.test_dataloader():
        # Evaluate on test set
        pass
""")


#*****************************************************#
#   part 8: pre-pipeline - performance optimization   #
#*****************************************************#
print("\n" + "="*80)
print(" PART 8: PRE-PIPELINE FOR FASTER TRAINING")
print("="*80)
print("""
ADVANCED FEATURE: Pre-Pipeline! 🚀

PROBLEM:
--------
Some operations are EXPENSIVE (coil combination, raw processing) but don't
need randomization. Running them every batch wastes time!

SOLUTION: Pre-Pipeline
-----------------------
Pre-pipeline runs ONCE during initialization and caches results.
Then fast augmentations (phase, noise, etc.) run on-the-fly.

SPEEDUP: 5-10x faster training! ⚡

Let's see it:
""")

print("\nEXAMPLE: Training with Pre-Pipeline")
print("-" * 60)

aug_with_prepipeline = Augmentrum(
    data=data_list,
    water=water_list,

    split_fractions={'val': 0.2, 'test': 0.1},
    seed=42,

    # PRE-PIPELINE: Expensive ops run ONCE and are cached
    pre_pipeline=['coil_sampling', 'processing'],

    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,

    # MAIN PIPELINES: Fast ops run on-the-fly
    pipelines={
        'train': ['phase', 'line_broadening', 'noise'],
        'val': ['phase'],
        'test': None
    },

    modes={
        'train': 'on-the-fly',
        'val': 'fixed',
        'test': 'fixed'
    },

    # Pre-pipeline params (sampled once during init)
    n_coils=8,
    n_averages=32,

    # Main pipeline params (sampled every batch)
    zero_order_deg=(-45, 45),
    lb_hz=(0.5, 3.0),
    sigma_frac=(0.01, 0.03),

    batch_size=16,
    backend='pytorch',
    volatile=True
)

print("\n✓ Created augmenter with pre-pipeline!")
print(f"\nPRE-PIPELINE (cached, runs once):")
print(f"  • coil_sampling - select and combine coils")
print(f"  • processing - process raw FID")

print(f"\nMAIN PIPELINE (on-the-fly, runs every batch):")
print(f"  • phase - random phase shifts")
print(f"  • line_broadening - random broadening")
print(f"  • noise - random noise")

print("""
PERFORMANCE:
  Without pre-pipeline: ~20-30 ms/batch
  With pre-pipeline:    ~2-5 ms/batch
  
  SPEEDUP: 5-10x faster! 🚀

PER-SPLIT PRE-PIPELINES (Advanced):
------------------------------------
You can also use different pre-pipelines per split:

aug = Augmentrum(
    ...
    pre_pipelines={
        'train': ['coil_sampling', 'processing'],  # Random coils
        'val': ['processing'],                      # Just processing
        'test': None                                # No preprocessing
    },
    ...
)

This gives maximum flexibility for different training strategies!
""")


#*************#
#   summary   #
#*************#
print("\n" + "="*80)
print(" 🎉 TUTORIAL 02 COMPLETE!")
print("="*80)
print("""
CONGRATULATIONS! You now understand:

✓ How to split data into train/val/test sets
✓ Why different splits need different augmentation strategies
✓ How to assign different pipelines to each split
✓ The critical difference between on-the-fly and fixed modes
✓ How to use split-specific dataloaders
✓ What provenance logging is and when to use volatile mode
✓ How to set up a complete training pipeline
✓ How to use pre-pipeline for 5-10x speedup! ← NEW!

KEY CONCEPTS:
-------------
1. SPLITS: Divide subjects into train/val/test
2. PIPELINES: Different for each split (heavy/light/none)
3. MODES: on-the-fly for training, fixed for val/test
4. DATALOADERS: Generate batches for each split
5. VOLATILE: Disable logging for speed in training
6. PRE-PIPELINE: Cache expensive ops for massive speedup ← NEW!

BEST PRACTICES CHECKLIST:
-------------------------
✓ Split data at subject level (not spectrum level)
✓ Use 70-80% for training, 10-20% for val, 10-20% for test
✓ Heavy augmentation for training (multiple modules + ranges)
✓ Light/no augmentation for validation (consistent evaluation)
✓ No augmentation for testing (unbiased evaluation)
✓ mode='on-the-fly' for training
✓ mode='fixed' for validation and testing
✓ volatile=True for training (speed)
✓ volatile=False when creating datasets to share
✓ Use seed for reproducible splits
✓ Use pre_pipeline for expensive operations ← NEW!

NEXT TUTORIAL:
--------------
• 03_backends_and_training.py - Multi-framework support and real training

READY TO TRAIN?
---------------
You now have everything you need to set up proper data augmentation for
training robust ML models on MRS data!

Copy the complete example from Part 7 and adapt it to your needs.

Happy training! 🚀
""")



