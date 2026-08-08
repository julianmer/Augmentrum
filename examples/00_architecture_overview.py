"""
================================================================================
        AUGMENTRUM ARCHITECTURE: DATA FLOW & PIPELINE DESIGN
================================================================================

This document provides a BIRD'S EYE VIEW of how Augmentrum works internally.


PIPELINE ARCHITECTURE:
======================

┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGE 0: DATA LOADING                            │
│                         (Happens ONCE)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Scanner Files (.dat, .sdat, .twix, .nii.gz)                            │
│         │                                                               │
│         ├─> Dataset Loader (COWSDataModule, etc.)                       │
│         │                                                               │
│         └─> List of NIfTI_MRS objects                                   │
│             • Contains raw FID data                                     │
│             • Contains metadata (scanner params, etc.)                  │
│             • May have multiple coils/averages                          │
│             • SVS: single voxel data                                    │
│             • MRSI: 2D/3D volumetric data                               │
│                                                                         │
│  OUTPUT: data_list (List[NIfTI_MRS])                                    │
│          water_list (List[NIfTI_MRS], optional)                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: AUGMENTRUM INITIALIZATION                    │
│                         (Setup phase)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  augmenter = Augmentrum(                                                │
│      data=data_list,         # Raw NIfTI objects                        │
│      water=water_list,                                                  │
│      pipeline=[...],         # Define processing steps                  │
│      backend='pytorch',      # Target format                            │
│      mode='on-the-fly',      # Random or fixed params                   │
│      ...                                                                │
│  )                                                                      │
│                                                                         │
│  What happens:                                                          │
│  • Data wrapped in NIfTI_MRS_Plus container                             │
│  • Pipeline objects created from module names/instances                 │
│  • Train/val/test splits defined (if requested)                         │
│  • NO processing yet - just setup!                                      │
│                                                                         │
│  OUTPUT: Augmentrum object (ready to generate batches)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: BATCH GENERATION LOOP                       │
│                    (Happens every iteration)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for batch_data, batch_water in augmenter.dataloader():                 │
│      # This loop triggers the pipeline for each batch                   │
│                                                                         │
│  For each batch:                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 2.1: SUBJECT SAMPLING                                            │   │
│  │      • Select batch_size subjects (random or sequential)         │   │
│  │      • Each subject is a NIfTI_MRS object                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 2.2: PRE-PIPELINE                                                │   │
│  │      • Optional fixed preprocessing                              │   │
│  │      • Applied BEFORE on-the-fly augmentation                    │   │
│  │      • Example: coil combination, raw processing                 │   │
│  │      • These steps run ONCE and are cached                       │   │
│  │      • Massive speedup for expensive operations!                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 2.3: MAIN PIPELINE (The Core Processing)                         │   │
│  │                                                                  │   │
│  │  BACKEND HANDLING (IMPORTANT!):                                  │   │
│  │  • You select a backend at initialization (e.g., 'pytorch')      │   │
│  │  • Pipeline tries to use YOUR backend for ALL modules            │   │
│  │  • If a module doesn't support your backend:                     │   │
│  │    1. Converts to module's preferred backend                     │   │
│  │    2. Runs the module                                            │   │
│  │    3. Converts BACK to your requested backend                    │   │
│  │  • This happens AUTOMATICALLY - you don't do anything!           │   │
│  │                                                                  │   │
│  │      Subject Data (NIfTI_MRS_Plus, backend='pytorch')            │   │
│  │              │                                                   │   │
│  │              ├─> Module 1 (e.g., PhaseShift)                     │   │
│  │              │   • Supports backend='pytorch'                    │   │
│  │              │   • Runs directly on pytorch tensors              │   │
│  │              │   • Samples random phase (if range specified)     │   │
│  │              │   • Output: pytorch tensor                        │   │
│  │              │                                                   │   │
│  │              ├─> Module 2 (e.g., NIfTI_RawProcessor)             │   │
│  │              │   • Only supports backend='nifti_list'            │   │
│  │              │   • CONVERTS: pytorch → nifti_list                │   │
│  │              │   • Processes raw FID data                        │   │
│  │              │   • CONVERTS BACK: nifti_list → pytorch           │   │
│  │              │   • Output: pytorch tensor (your requested!)      │   │
│  │              │                                                   │   │
│  │              ├─> Module 3 (e.g., LineBroadening)                 │   │
│  │              │   • Supports backend='pytorch'                    │   │
│  │              │   • Runs directly on pytorch tensors              │   │
│  │              │   • Samples random broadening (if range)          │   │
│  │              │   • Output: pytorch tensor                        │   │
│  │              │                                                   │   │
│  │              └─> Module 4 (e.g., Noise)                  │   │
│  │                  • Supports backend='pytorch'                    │   │
│  │                  • Runs directly on pytorch tensors              │   │
│  │                  • Samples random noise level (if range)         │   │
│  │                  • Output: pytorch tensor                        │   │
│  │                                                                  │   │
│  │      Pipeline Output: Augmented data in YOUR backend (pytorch)   │   │
│  │                                                                  │   │
│  │  KEY POINT: Data stays in your requested backend as much as      │   │
│  │             possible. Conversions are MINIMIZED automatically!   │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 2.5: POST-PIPELINE (User-Defined, Optional)                      │   │
│  │      • Additional processing AFTER dataloader                    │   │
│  │      • User can add their own transforms                         │   │
│  │      • Example: normalization, FFT, etc.                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  OUTPUT: (batch_data, batch_water)                                      │
│          • batch_data: Tensor/array of shape (batch_size, n_points)     │
│          • batch_water: Tensor/array of shape (batch_size, n_points)    │
│          • Ready for neural network!                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: NEURAL NETWORK                            │
│                     (Your training code)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for epoch in range(num_epochs):                                        │
│      for batch_data, batch_water in augmenter.train_dataloader():       │
│          # batch_data is already a tensor!                              │
│          batch_data = batch_data.to('cuda')  # Move to GPU              │
│          outputs = model(batch_data)         # Forward pass             │
│          loss = criterion(outputs, targets)                             │
│          loss.backward()                     # Backward pass            │
│          optimizer.step()                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


BACKEND SYSTEM EXPLAINED:
==========================

HOW BACKEND SELECTION WORKS:
-----------------------------

When you create an Augmentrum instance, you specify a backend:

    aug = Augmentrum(data=data_list, backend='pytorch', ...)

This backend becomes the PREFERRED format for ALL operations. Here's what happens:

┌─────────────────────────────────────────────────────────────────────────┐
│ BACKEND FLOW IN PIPELINE                                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ 1. DATA ENTERS PIPELINE                                                 │
│    • Format: NIfTI_MRS_Plus with backend='pytorch'                      │
│    • Contains tensor data internally                                    │
│                                                                         │
│ 2. FOR EACH MODULE:                                                     │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │ Check: Does module support 'pytorch' backend?               │      │
│    └─────────────────────────────────────────────────────────────┘      │
│                        ↓                       ↓                        │
│                       YES                     NO                        │
│                        │                       │                        │
│           ┌────────────┴──────────┐   ┌────────┴───────────┐            │
│           │ Use pytorch directly  │   │ Need conversion    │            │
│           │ • Fast!               │   │ • Slower           │            │
│           │ • No overhead         │   │ • But automatic    │            │
│           └────────────┬──────────┘   └─────────┬──────────┘            │
│                        │                        │                       │
│                        │               ┌────────┴──────────┐            │
│                        │               │ Convert to        │            │
│                        │               │ module's backend  │            │
│                        │               │ (e.g., nifti_list)│            │
│                        │               └────────┬──────────┘            │
│                        │                        │                       │
│                        │               ┌────────┴─────────┐             │
│                        │               │ Run module       │             │
│                        │               └────────┬─────────┘             │
│                        │                        │                       │
│                        │               ┌────────┴──────────┐            │
│                        │               │ Convert BACK to   │            │
│                        │               │ pytorch           │            │
│                        │               └────────┬──────────┘            │
│                        │                        │                       │
│                        └────────────────────────┘                       │
│                                    │                                    │
│                        ┌───────────┴──────────┐                         │
│                        │ Data still pytorch   │                         │
│                        │ for next module      │                         │
│                        └───────────┬──────────┘                         │
│                                    │                                    │
│ 3. PIPELINE OUTPUT                                                      │
│    • Format: NIfTI_MRS_Plus with backend='pytorch'                      │
│    • Ready for your training code!                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""