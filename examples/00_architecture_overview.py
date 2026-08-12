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
│  │              ├─> Module 2 (e.g., RawProcessor)             │   │
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
│  │              └─> Module 4 (e.g., Noise)                          │   │
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


DOMAIN SYSTEM EXPLAINED:
=========================

MRS data lives in one of two domains per axis: spectral (time FID vs.
frequency spectrum) and spatial (image vs. k-space). A module's math is only
correct in one of them — line broadening multiplies a decay onto the FID, a
baseline is added onto the spectrum, an undersampling mask zeros k-space
bins. Handed the wrong domain, a module still returns a number; just the
wrong one, silently.

HOW MODULES HANDLE THIS:
------------------------

Every module DECLARES where its math works — it never transforms internally:

    LineBroadening.DOMAIN        -> Domain(spectral='time')
    BaselineAugmentation.DOMAIN  -> Domain(spectral='frequency')
    KspaceUndersampling.DOMAIN   -> Domain(spatial='kspace')   # mask modes
    Noise.DOMAIN                 -> None (white noise works anywhere)

A DOMAIN can even depend on the sampled parameters: PhaseShift needs a
spectrum only when a first-order ramp is in play, so its DOMAIN follows the
value that actually runs — including per-batch range sampling.

┌─────────────────────────────────────────────────────────────────────────┐
│ DOMAIN FLOW IN PIPELINE (domain_planning='auto', the default)           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  pipeline = ['line_broadening', 'baseline', 'residual_water', 'noise']  │
│                                                                         │
│  The planner walks the chain ONCE, tracking where the data is:          │
│                                                                         │
│      data (time) ──> LineBroadening          needs time      ✓ no move  │
│                 ┌──> [DomainTransform to frequency]          INSERTED   │
│                 ├──> BaselineAugmentation    needs frequency ✓          │
│                 ├──> ResidualWater           needs frequency ✓ SHARED!  │
│                 ├──> Noise                   needs nothing   ✓          │
│                 └──> [DomainTransform to time]               INSERTED   │
│                                                                         │
│  • Transforms are inserted ONLY where the declared domain is not        │
│    already satisfied — a run of same-domain modules shares one move.    │
│  • DomainTransforms YOU place in the pipeline count: a correctly        │
│    hand-placed chain gets nothing added; a misplaced one is fixed.      │
│  • The data is left in time/image at the end so it can be written out   │
│    (override with end_domain to stay in k-space, for instance).         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

TURNING THE AUTOMATION OFF:
---------------------------

If you place every transform yourself and want mistakes SURFACED instead of
silently fixed:

    Augmentrum(..., domain_planning='strict')
    # or AugmentationPipeline([...], domain_planning='strict')

Strict planning inserts nothing; any module reached in the wrong domain
raises a DomainError telling you exactly where to add a DomainTransform.
A module can also declare STRICT = True to refuse being moved individually.

WHY MODULES NEVER TRANSFORM INTERNALLY:
---------------------------------------

A hidden fft -> op -> ifft inside a module would cost a redundant round-trip
every time neighboring modules want the same domain, and would double-
transform data that is already there. Declaring the domain instead makes
transforms explicit, shared, and — through the planner — provably minimal.
(The one exception: process_nifti_list paths convert at the format boundary,
because the NIfTI-MRS format itself stores FIDs.)
"""