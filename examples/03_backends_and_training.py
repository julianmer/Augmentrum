"""
================================================================================
        AUGMENTRUM TUTORIAL 03: BACKENDS, EXPORT & REAL TRAINING
================================================================================

📖 PREREQUISITE: Complete Tutorials 01 & 02 first!
   For data flow diagrams, see 00_architecture_overview.py

WHAT YOU'LL LEARN:
==================
In this final tutorial, we'll cover advanced topics for integrating Augmentrum
with machine learning frameworks and production workflows:

• What backends are and why they matter
• How to use numpy, PyTorch, TensorFlow, and JAX backends
• Automatic backend conversions
• How to export augmented data to NIfTI files
• Complete training example with PyTorch
• Best practices for each framework

WHAT ARE BACKENDS?
==================
A "backend" is the data format used internally by Augmentrum. Different ML
frameworks expect different formats:

┌─────────────────────────────────────────────────────────────┐
│ NIFTI_LIST Backend                                          │
│ • Data: List of NIfTI_MRS objects                           │
│ • Use for: FSL-MRS compatibility, metadata preservation     │
│ • Export: Save as .nii.gz files                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ NUMPY Backend                                               │
│ • Data: numpy.ndarray                                       │
│ • Use for: Prototyping, custom algorithms, visualization    │
│ • Fast for CPU-only work                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PYTORCH Backend                                             │
│ • Data: torch.Tensor                                        │
│ • Use for: Training neural networks with PyTorch            │
│ • Supports GPU acceleration                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TENSORFLOW Backend                                          │
│ • Data: tf.Tensor                                           │
│ • Use for: Training with TensorFlow/Keras                   │
│ • Supports GPU/TPU acceleration                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ JAX Backend                                                 │
│ • Data: jax.Array                                           │
│ • Use for: Training with JAX                                │
│ • JIT compilation and GPU support                           │
└─────────────────────────────────────────────────────────────┘

THE MAGIC: AUTOMATIC CONVERSION!
=================================
Augmentrum automatically converts between backends as needed. You can:

• Store data internally as 'numpy'
• Request batches as 'pytorch' tensors
• Augmentrum converts automatically!

This makes it incredibly flexible for different workflows.

Let's explore!
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from augmentrum import Augmentrum, Backend
from augmentrum.dataset.cows import COWSDataModule

# Plotting
import matplotlib.pyplot as plt
import numpy as np


# Simple plotting helper using FSL-MRS plot_spectra for overlay
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
            if dim is None:
                continue
            if processed.shape[processed.dim_position(dim)] > 1:
                if dim in ('DIM_EDIT', 'DIM_METCYCLE', 'DIM_ISIS'):
                    processed = nifti_mrs_proc.subtract(processed, dim=dim)
                else:
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

    # Use FSL-MRS plot_spectra to overlay in ONE figure
    fig = plot_spectra(mrs_list, ppmlim=ppmlim, legend=True)
    
    # Update with custom title and labels
    if fig and len(fig.axes) > 0:
        ax = fig.axes[0]
        ax.set_title(title, fontsize=12, fontweight='bold')
        # Update legend with custom labels
        if len(labels) == len(mrs_list):
            ax.legend(labels, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    return fig


#**************************#
#   part 1: loading data   #
#**************************#
print("\n" + "="*80)
print(" PART 1: LOADING DATA")
print("="*80)

data_dir = os.environ.get(
    "DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openneuro_ds006812"))
)

print(f"Loading COWS dataset from: {data_dir}\n")

def load_cows_data(data_dir, location='PARIETAL', water_sup='VAPOR'):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data not found: {data_dir}")
    loader = COWSDataModule(data_dir=data_dir, location=location, water_sup=water_sup)
    data_list, water_list, _, _, names = loader.load_twix()
    if len(data_list) == 0:
        raise RuntimeError("No data loaded.")
    return data_list, water_list, names

data_list, water_list, subject_names = load_cows_data(data_dir)
print(f"✓ Loaded {len(data_list)} subjects\n")


#************************************#
#   part 2: understanding backends   #
#************************************#
print("="*80)
print(" PART 2: EXPLORING DIFFERENT BACKENDS")
print("="*80)
print("""
Let's create the same augmenter with different backends and see what we get.
The augmentation is identical - only the output format changes!
""")

# Common pipeline
common_pipeline = ['processing', 'phase', 'noise']
common_params = {
    'zero_order_deg': 30.0,
    'sigma_frac': 0.02,
    'batch_size': 2,
    'mode': 'fixed'
}

print("\nBACKEND 1: NUMPY")
print("-" * 60)

aug_numpy = Augmentrum(
    data=data_list[:3],
    pipeline=common_pipeline,
    backend='numpy',  # ← numpy backend
    **common_params
)

batch_np, _ = next(aug_numpy.dataloader(framework='numpy'))
print(f"✓ NumPy backend")
print(f"  Type: {type(batch_np)}")
print(f"  Shape: {batch_np.shape}")
print(f"  Dtype: {batch_np.dtype}")
print(f"  Device: CPU (numpy is CPU-only)")
print(f"\n  Use cases:")
print(f"  • Prototyping and quick experiments")
print(f"  • Custom algorithms")
print(f"  • Visualization")
print(f"  • Integration with scikit-learn")

print("\n\nBACKEND 2: PYTORCH")
print("-" * 60)

try:
    import torch

    aug_pytorch = Augmentrum(
        data=data_list[:3],
        pipeline=common_pipeline,
        backend='pytorch',  # ← PyTorch backend
        **common_params
    )

    batch_pt, _ = next(aug_pytorch.dataloader(framework='pytorch'))
    print(f"✓ PyTorch backend")
    print(f"  Type: {type(batch_pt)}")
    print(f"  Shape: {batch_pt.shape}")
    print(f"  Dtype: {batch_pt.dtype}")
    print(f"  Device: {batch_pt.device}")
    print(f"\n  Use cases:")
    print(f"  • Training neural networks with PyTorch")
    print(f"  • GPU acceleration (can move to CUDA)")
    print(f"  • PyTorch ecosystem (torchvision, etc.)")

    # Show GPU capability
    if torch.cuda.is_available():
        batch_gpu = batch_pt.to('cuda')
        print(f"\n  ✓ Can move to GPU: {batch_gpu.device}")
    else:
        print(f"\n  (No GPU available, but would work with CUDA)")

except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    print("Skipping PyTorch examples...")

print("\n\nBACKEND 3: NIFTI_LIST")
print("-" * 60)

aug_nifti = Augmentrum(
    data=data_list[:3],
    water=water_list[:3] if water_list else None,
    pipeline=common_pipeline,
    backend='nifti_list',  # ← NIfTI backend
    **common_params
)

batch_nifti, _ = next(aug_nifti.dataloader())
print(f"✓ NIfTI_list backend")
print(f"  Type: {type(batch_nifti)} (list)")
print(f"  Number of spectra: {len(batch_nifti)}")
print(f"  Each element: {type(batch_nifti[0])}")
print(f"\n  Use cases:")
print(f"  • FSL-MRS compatibility")
print(f"  • Metadata preservation")
print(f"  • Exporting to NIfTI files")
print(f"  • Integration with FSL/Osprey/TARQUIN")


#******************************************#
#   part 3: automatic backend conversion   #
#******************************************#
print("\n" + "="*80)
print(" PART 3: AUTOMATIC BACKEND CONVERSION - THE MAGIC!")
print("="*80)
print("""
Here's the cool part: You can store data in one backend but request batches
in a different backend. Augmentrum converts automatically!

EXAMPLE:
  • Store internally as 'numpy' (lightweight)
  • Request batches as 'pytorch' (for training)
  • Conversion happens automatically!

This is incredibly flexible. Let's see it in action:
""")

print("\nEXAMPLE: NumPy storage, PyTorch output")
print("-" * 60)

try:
    import torch

    # Create augmenter with numpy backend
    aug_convert = Augmentrum(
        data=data_list[:3],
        water=water_list[:3] if water_list else None,
        pipeline=['processing', 'noise'],
        backend='numpy',  # ← Store as numpy
        sigma_frac=0.02,
        coil_method='fsl-mrs',
        conj=False,
        batch_size=2,
        mode='fixed'
    )

    print("Created augmenter with backend='numpy'")

    # Request batches as PyTorch tensors!
    batch_torch, _ = next(aug_convert.dataloader(framework='pytorch'))

    print(f"\n✓ Requested PyTorch batches from numpy backend")
    print(f"  Returned type: {type(batch_torch)}")
    print(f"  Shape: {batch_torch.shape}")
    print(f"  Can use directly in PyTorch model!")

    # You can also request numpy from pytorch backend
    aug_pt = Augmentrum(
        data=data_list[:3],
        water=water_list[:3] if water_list else None,
        pipeline=['processing'],
        backend='pytorch',  # ← Store as pytorch
        coil_method='fsl-mrs',
        conj=False,
        batch_size=2,
        mode='fixed'
    )

    batch_np, _ = next(aug_pt.dataloader(framework='numpy'))
    print(f"\n✓ Requested NumPy batches from pytorch backend")
    print(f"  Returned type: {type(batch_np)}")
    print(f"  Shape: {batch_np.shape}")

    print("""
    KEY POINT:
    The backend parameter sets INTERNAL storage format.
    The framework parameter in dataloader() sets OUTPUT format.
    Augmentrum handles the conversion!
    """)

except ImportError:
    print("PyTorch not available for this demo")


#**************************************#
#   part 4: exporting to nifti files   #
#**************************************#
print("\n" + "="*80)
print(" PART 4: EXPORTING AUGMENTED DATA TO NIFTI FILES")
print("="*80)
print("""
Sometimes you want to save augmented data for later use or to share with
collaborators. Augmentrum makes this easy!

EXPORT WORKFLOW:
----------------
1. Use backend='nifti_list' (preserves NIfTI format)
2. Set volatile=False (keeps metadata/provenance)
3. Generate batches
4. Save each NIfTI object to .nii.gz file

Let's create an augmented dataset and export it:
""")

# Create temporary directory for demonstration
temp_dir = tempfile.mkdtemp(prefix="augmentrum_export_")
print(f"\n✓ Created temporary export directory: {temp_dir}")

# Create augmenter for export (use subset for demo)
export_data = data_list[:3]  # Just 3 subjects for demo
export_water = water_list[:3] if water_list else None

aug_export = Augmentrum(
    data=export_data,
    water=export_water,
    pipeline=['processing', 'phase', 'line_broadening', 'noise'],
    zero_order_deg=30.0,
    lb_hz=2.0,
    sigma_frac=0.02,
    coil_method='fsl-mrs',
    conj=False,
    backend='nifti_list',  # ← Keep NIfTI format
    volatile=False,        # ← Keep provenance!
    batch_size=3,
    mode='fixed'
)

print("\nGenerating augmented batch...")
batch_data, batch_water = next(aug_export.dataloader())

print(f"\n✓ Generated {len(batch_data)} augmented spectra")
print("\nExporting to NIfTI files...")

# When framework='python', batch_data is already a list of NIfTI_MRS objects
# If it's a NIfTI_MRS_Plus, we need to call .list()
if hasattr(batch_data, 'list'):
    nifti_list = batch_data.list()
else:
    nifti_list = batch_data

# Export metabolite spectra
for i, nifti in enumerate(nifti_list):
    output_path = os.path.join(temp_dir, f"augmented_metab_{i:03d}.nii.gz")
    nifti.save(output_path)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")

# Export water references if available
if batch_water is not None:
    if hasattr(batch_water, 'list'):
        exported_water_list = batch_water.list()
    else:
        exported_water_list = batch_water

    for i, nifti in enumerate(exported_water_list):
        output_path = os.path.join(temp_dir, f"augmented_water_{i:03d}.nii.gz")
        nifti.save(output_path)
        print(f"  ✓ Saved: {os.path.basename(output_path)}")

# List all exported files
all_files = sorted(os.listdir(temp_dir))
print(f"\n✓ Export complete!")
print(f"  Total files: {len(all_files)}")
print(f"  Location: {temp_dir}")
print(f"\n  These .nii.gz files can be loaded in:")
print(f"  • FSL (for processing/analysis)")
print(f"  • Osprey (for quantification)")
print(f"  • TARQUIN (for fitting)")
print(f"  • Any NIfTI-compatible software")

# Visualize one exported spectrum
print(f"\nLoading and visualizing one exported spectrum...")
from nifti_mrs.nifti_mrs import NIFTI_MRS

first_file = os.path.join(temp_dir, all_files[0])
loaded_nifti = NIFTI_MRS(first_file)
print(f"  ✓ Loaded: {all_files[0]}")
print(f"  Shape: {loaded_nifti.shape}")
print(f"  Metadata preserved: Yes")

# Use native NIFTI_MRS plotting
loaded_nifti.plot(ppmlim=(0.2, 4.2), legend=True)
plt.gcf().suptitle(f"Exported Spectrum: {all_files[0]}", fontsize=12, fontweight='bold')
plt.show()

# Cleanup
print(f"\nCleaning up temporary directory...")
shutil.rmtree(temp_dir)
print(f"  ✓ Removed: {temp_dir}")


#***********************************************#
#   part 5: complete pytorch training example   #
#***********************************************#
print("\n" + "="*80)
print(" PART 5: COMPLETE PYTORCH TRAINING EXAMPLE")
print("="*80)
print("""
Now let's put it all together with a REAL PyTorch training example!

We'll create a simple neural network and train it on augmented MRS data.
This demonstrates the complete workflow from data loading to training.

ARCHITECTURE:
-------------
We'll build a simple 1D CNN for demonstration:
  • Input: MRS spectrum (4096 points)
  • Conv1D layers to extract features
  • Fully connected layers for prediction
  • Output: Some prediction (e.g., metabolite concentrations)

For this demo, we'll create synthetic targets (in real use, you'd have
actual labels like metabolite concentrations, disease status, etc.)
""")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print("\n✓ PyTorch is available. Starting training example...")

    #******************************#
    #   step 1: define the model   #
    #******************************#
    print("\n" + "-"*60)
    print("STEP 1: Define Neural Network Architecture")
    print("-"*60)

    class MRSNet(nn.Module):
        """
        Simple 1D CNN for MRS data.

        Architecture:
          Input (4096) → Conv1D → Pool → Conv1D → Pool → Flatten → FC → Output
        """
        def __init__(self, input_size=4096, num_outputs=5):
            super(MRSNet, self).__init__()

            self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            self.pool = nn.MaxPool1d(4)
            self.relu = nn.ReLU()

            # Calculate flattened size after convolutions
            conv_output_size = input_size // (4 * 4)  # Two pooling layers
            fc_input_size = 32 * conv_output_size

            self.fc1 = nn.Linear(fc_input_size, 128)
            self.fc2 = nn.Linear(128, num_outputs)

        def forward(self, x):
            # x shape: (batch_size, n_points)
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, n_points)

            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)

            x = x.view(x.size(0), -1)  # Flatten
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

            return x

    model = MRSNet(input_size=4096, num_outputs=5)
    print(f"\n✓ Created model: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    #*************************************#
    #   step 2: setup data augmentation   #
    #*************************************#
    print("\n" + "-"*60)
    print("STEP 2: Setup Data Augmentation")
    print("-"*60)

    augmenter_train = Augmentrum(
        data=data_list,
        water=water_list,

        # Splits
        split_fractions={'val': 0.2},
        seed=42,

        # Pipelines
        pipelines={
            'train': ['processing', 'phase', 'line_broadening', 'noise'],
            'val': ['processing']
        },

        # Modes
        modes={
            'train': 'on-the-fly',  # Different every epoch!
            'val': 'fixed'          # Same for consistent eval
        },

        # Augmentation parameters
        zero_order_deg=(-45, 45),
        lb_hz=(0.5, 3.0),
        sigma_frac=(0.005, 0.03),

        # Settings for PyTorch
        batch_size=8,
        backend='pytorch',  # ← PyTorch backend!
        volatile=True       # Fast training
    )

    print(f"\n✓ Created augmenter")
    print(f"  Train: {len(augmenter_train.splits['train'][0])} subjects")
    print(f"  Val:   {len(augmenter_train.splits['val'][0])} subjects")
    print(f"  Batch size: {augmenter_train.batch_size}")

    #**************************************#
    #   step 3: create synthetic targets   #
    #**************************************#
    print("\n" + "-"*60)
    print("STEP 3: Create Synthetic Targets")
    print("-"*60)
    print("(In real use, you'd have actual labels)")

    # Create random targets for demonstration
    # In real use, these would be metabolite concentrations, disease labels, etc.
    num_subjects = len(data_list)
    synthetic_targets = torch.randn(num_subjects, 5)  # 5 output values

    print(f"\n✓ Created targets: shape={synthetic_targets.shape}")

    #****************************#
    #   step 4: training setup   #
    #****************************#
    print("\n" + "-"*60)
    print("STEP 4: Training Setup")
    print("-"*60)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n✓ Loss function: {criterion.__class__.__name__}")
    print(f"  Optimizer: {optimizer.__class__.__name__}")
    print(f"  Learning rate: 0.001")

    #***************************#
    #   step 5: training loop   #
    #***************************#
    print("\n" + "-"*60)
    print("STEP 5: Training (3 epochs for demonstration)")
    print("-"*60)

    num_epochs = 3

    for epoch in range(num_epochs):
        #********************#
        #   training phase   #
        #********************#
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (batch_data, _) in enumerate(augmenter_train.train_dataloader()):
            # batch_data is already a torch.Tensor from pytorch backend!
            # Shape: (batch_size, 1, 1, 1, 1, 1, n_points) -> reshape to (batch_size, n_points)
            # Keep first dimension (batch) and last dimension (spectral points), squeeze middle
            batch_data = batch_data.view(batch_data.shape[0], -1)

            # Get corresponding targets
            # (In real use, targets would be associated with subjects)
            batch_size = batch_data.shape[0]
            batch_targets = synthetic_targets[:batch_size]

            # Forward pass
            outputs = model(batch_data.float())
            loss = criterion(outputs, batch_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Limit batches for demo
            if train_batches >= 5:
                break

        avg_train_loss = train_loss / train_batches

        #**********************#
        #   validation phase   #
        #**********************#
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (batch_data, _) in enumerate(augmenter_train.val_dataloader()):
                # Reshape batch_data to (batch_size, n_points)
                batch_data = batch_data.view(batch_data.shape[0], -1)
                
                batch_size = batch_data.shape[0]
                batch_targets = synthetic_targets[:batch_size]

                outputs = model(batch_data.float())
                loss = criterion(outputs, batch_targets)

                val_loss += loss.item()
                val_batches += 1

                # Limit batches for demo
                if val_batches >= 2:
                    break

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

    print(f"\n✓ Training complete!")
    print(f"\nKEY POINTS:")
    print(f"  • Augmentation happens automatically during data loading")
    print(f"  • Training sees DIFFERENT augmentations each epoch")
    print(f"  • Validation sees SAME data each epoch (for consistent metrics)")
    print(f"  • No manual conversion needed - backend='pytorch' handles it!")

except ImportError:
    print("\nPyTorch not installed.")
    print("Install with: pip install torch")
    print("\nHere's what the training code would look like:")
    print("""
import torch
import torch.nn as nn

# Create model
model = YourModel()

# Setup augmentation
augmenter = Augmentrum(
    data=data_list,
    water=water_list,
    split_fractions={'val': 0.2},
    pipelines={
        'train': ['processing', 'phase', 'noise'],
        'val': ['processing']
    },
    modes={'train': 'on-the-fly', 'val': 'fixed'},
    backend='pytorch',
    batch_size=32
)

# Training loop
for epoch in range(num_epochs):
    for batch, _ in augmenter.train_dataloader():
        loss = model(batch)
        loss.backward()
        optimizer.step()
""")


#************************************#
#   part 6: best practices summary   #
#************************************#
print("\n" + "="*80)
print(" PART 6: BEST PRACTICES BY FRAMEWORK")
print("="*80)
print("""
Here's a quick reference for using Augmentrum with different frameworks:

PYTORCH:
--------
• backend='pytorch'
• Request batches with framework='pytorch' (or default)
• Move to GPU: batch.to('cuda')
• Use in training loops directly

Example:
  aug = Augmentrum(data=data, backend='pytorch', ...)
  for batch, _ in aug.train_dataloader():
      batch = batch.to('cuda')
      output = model(batch)

NUMPY / SCIKIT-LEARN:
---------------------
• backend='numpy'
• Request batches with framework='numpy'
• Perfect for prototyping and classical ML

Example:
  aug = Augmentrum(data=data, backend='numpy', ...)
  for batch, _ in aug.dataloader(framework='numpy'):
      predictions = sklearn_model.predict(batch)

TENSORFLOW / KERAS:
-------------------
• backend='tensorflow'
• Request batches with framework='tensorflow'
• Use with tf.data.Dataset for efficiency

Example:
  aug = Augmentrum(data=data, backend='tensorflow', ...)
  for batch, _ in aug.dataloader(framework='tensorflow'):
      output = model(batch)

EXPORTING DATA:
---------------
• backend='nifti_list'
• volatile=False (keep metadata)
• Save with nifti.save('file.nii.gz')

Example:
  aug = Augmentrum(data=data, backend='nifti_list', volatile=False, ...)
  for batch, _ in aug.dataloader():
      for i, nifti in enumerate(batch):
          nifti.save(f'output_{i}.nii.gz')

GENERAL TIPS:
-------------
✓ Use volatile=True for training (faster)
✓ Use volatile=False when exporting (preserves metadata)
✓ Choose backend based on your framework
✓ Let Augmentrum handle conversions automatically
✓ Use batch_size appropriate for your GPU memory
✓ mode='on-the-fly' for training, mode='fixed' for eval
""")


#******************************#
#   part 7: supervised pairs   #
#******************************#
print("\n" + "="*80)
print(" PART 7: SUPERVISED PAIRS — TAPS AND OUTPUTS")
print("="*80)
print("""
Supervised training (reconstruction, denoising) needs (input, target) pairs.
Mark any pipeline stage with 'tap:<name>' and choose what the dataloaders
yield with the `outputs` spec — a nested tuple of stage tokens:

  'data' / 'water'              the pipeline end
  '<tap>' / '<tap>.water'       a tapped stage

Everything before the tap (properties of the object: macromolecules, line
broadening, phase) lands in BOTH input and target; everything after the tap
(undersampling, noise) degrades only the input.
""")

aug_pairs = Augmentrum(
    data=data_list,
    pipeline=['line_broadening', 'tap:clean', 'noise'],
    lb_hz=(0, 6),
    sigma_frac=(0.02, 0.05),
    outputs=(('data', 'water'), ('clean', 'clean.water')),
    batch_size=4,
    backend='pytorch',
    volatile=True,
)

(x, x_water), (y, y_water) = next(aug_pairs.dataloader())
print(f"✓ input  x: {tuple(x.shape)} — broadened AND noisy")
print(f"✓ target y: {tuple(y.shape)} — broadened only (frozen at the tap)")
print(f"  pairs differ: {not (x == y).all().item()}")
print("""
For a full MRSI reconstruction training example, see
examples/04_mrsi_recon_training.py — and scripts/train_deep_er.py for the
real thing (a Deep-ER-style network on the MRSI Challenge with faithful
ECCENTRIC undersampling).
""")


#*************#
#   summary   #
#*************#
print("\n" + "="*80)
print(" 🎉 TUTORIAL 03 COMPLETE - YOU'VE MASTERED AUGMENTRUM!")
print("="*80)
print("""
CONGRATULATIONS! You've completed all three tutorials and now understand:

✓ What backends are and how to use them
✓ How to work with numpy, PyTorch, TensorFlow backends
✓ Automatic backend conversion
✓ How to export augmented data to NIfTI files
✓ How to train a neural network with augmented data
✓ Best practices for each framework

YOU NOW KNOW:
-------------
TUTORIAL 01:
  • Augmentrum basics
  • Building pipelines
  • Fixed vs. random parameters
  • Creating custom modules

TUTORIAL 02:
  • Train/val/test splits
  • Split-specific pipelines
  • On-the-fly vs fixed modes
  • Using dataloaders

TUTORIAL 03:
  • Multi-framework support
  • Backend conversions
  • Exporting data
  • Complete training workflow

YOU'RE READY TO:
----------------
✓ Build custom augmentation pipelines
✓ Train robust ML models on MRS data
✓ Export augmented datasets
✓ Integrate with any ML framework
✓ Create your own modules
✓ Follow best practices

QUICK START TEMPLATE:
---------------------
Here's a complete template for your own projects:

from augmentrum import Augmentrum

# Your data
data_list = [...]  # Load your MRS data
water_list = [...]  # Load your water references

# Setup augmentation
aug = Augmentrum(
    data=data_list,
    water=water_list,
    
    # Splits
    split_fractions={'val': 0.15, 'test': 0.15},
    seed=42,
    
    # Pipelines
    pipelines={
        'train': ['processing', 'phase', 'line_broadening', 'noise'],
        'val': ['processing'],
        'test': None
    },
    
    # Modes
    modes={
        'train': 'on-the-fly',
        'val': 'fixed',
        'test': 'fixed'
    },
    
    # Processing parameters for COWS data
    coil_method='fsl-mrs',
    conj=False,
    
    # Parameters
    zero_order_deg=(-45, 45),
    lb_hz=(0.5, 3.0),
    sigma_frac=(0.005, 0.03),
    
    # Settings
    batch_size=32,
    backend='pytorch',  # or 'numpy', 'tensorflow'
    volatile=True
)

# Train your model
for epoch in range(num_epochs):
    for batch, water in aug.train_dataloader():
        # Your training code here
        pass

NEXT STEPS:
-----------
• Explore the augmentrum source code
• Try different augmentation strategies
• Experiment with custom modules
• Train your own models!
• Share your results with the community

Thank you for learning Augmentrum! 🚀

Questions or issues? Open an issue on GitHub!
Happy augmenting and happy training! 🎉
""")



