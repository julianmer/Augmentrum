<div align="center">
  <img src="https://raw.githubusercontent.com/julianmer/Augmentrum/main/assets/figures/logo.png" alt="Augmentrum Logo" width="100" style="margin-bottom: -10px;"/>
  <h1 style="margin-top: 5px; margin-bottom: 5px;">Augmentrum</h1>
  <p style="margin-top: 0px;"><em>A Data Augmentation Package for MR Spectroscopy</em></p>
  
  [![PyPI version](https://badge.fury.io/py/augmentrum.svg)](https://badge.fury.io/py/augmentrum)
  [![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![ISMRM 2026](https://img.shields.io/badge/ISMRM-Abstract%20%2305685-lightgrey.svg)](https://submissions.mirasmart.com/ISMRM2026/Itinerary/PresentationDetail.aspx?evdid=1450)
</div>

---

## Overview
**Augmentrum** is a modular Python framework designed to help researchers with limited *in-vivo* MRS data create diverse, physically consistent datasets through flexible augmentation. It supports **k-space resampling**, **coil and average sampling**, **signal-level perturbations**, and **synthetic artifact generation**, expanding both synthetic and *in-vivo* data in a realistic and controlled manner.

<div align="center">
  <img src="https://raw.githubusercontent.com/julianmer/Augmentrum/main/assets/figures/overview.png" alt="Augmentrum Pipeline Overview" width="800"/>
</div>

Built for data-driven MRS applications, Augmentrum streamlines the integration of data augmentation into existing workflows. It operates on the **NIfTI-MRS** standard, making it compatible with any acquisition format. Simply load your data as NIfTI (using `spec2nii` if needed) and apply either predefined augmentation settings or build a custom pipeline by combining modules. Each augmentation can be parameterized or used with default ranges to populate a dense and diverse dataset environment.

From MRSI reconstruction and acquisition variability to spectral perturbations and artifact synthesis, Augmentrum handles it all in a **modular**, **flexible**, and **easy-to-use** structure. Dataloaders allow **on-the-fly augmentation** for deep learning backends such as **PyTorch**, **TensorFlow**, **Keras**, and **JAX**, enabling robust training beyond static datasets.

> **Note:** Augmentrum is currently in **alpha development**. It is an active research framework and may undergo changes as modules and interfaces evolve.

---

## Features
- Modular augmentation across time, frequency, and k-space domains
- Physically valid transformations for realistic variability
- Native NIfTI-MRS I/O and metadata tracking
- Customizable pipelines with user-defined parameters
- On-the-fly augmentation for machine learning workflows

---

## Installation
```bash
pip install augmentrum
```
or from source:
```bash
git clone https://github.com/julianmer/Augmentrum.git
cd Augmentrum
pip install -e .
```

---

## Quick Start
```python
from augmentrum import Augmentrum

data   # list of NIfTI-MRS files
water   # list of corresponding water reference files (optional)

# initialize Augmentrum with data and optional water references
augmenter = Augmentrum(
    data=data, 
    water=water, 
    
    # custom pipeline example
    pipeline = [
        'coil_sampling',
        'average_sampling',
        'processing',
        'line_broadening',
        'baseline',
        'noise',
    ],
    
    # general settings
    batch_size=16,
    backend='pytorch',  # or 'tensorflow', 'keras', 'jax', 'numpy'
)

# get a dataloader for PyTorch with on-the-fly augmentation
train_data = augmenter.dataloader()
```

### Supervised Pairs: Taps and `outputs`

Any pipeline stage can be named with a `'tap:<name>'` marker and yielded by the
dataloaders through the `outputs` spec — an arbitrarily nested tuple of stage
tokens (`'data'` / `'water'` for the pipeline end, `'<tap>'` / `'<tap>.water'`
for a tapped stage). The classic `(data, water)` pair stays the default.

```python
augmenter = Augmentrum(
    data=data,
    pipeline=['coil_sampling', 'macromolecules', 'tap:clean', 'undersampling'],
    outputs=(('data', 'water'), ('clean', 'clean.water')),
    backend='pytorch',
)

# supervised reconstruction pairs: input fully augmented, target frozen at the tap
for (x, x_water), (y, y_water) in augmenter.dataloader():
    train_step(x, y)
```

---

## Module Reference & Backend Support

Every module is called as `module(nifti_plus, water)`.  When a backend is not
natively supported, the base class automatically routes the call through the
NIfTI-list path and returns the result in the original backend format —
no manual conversion needed.

| Module | Modes / Methods                                         | NIfTI | NumPy | PyTorch | TensorFlow | JAX | Keras |
|:---|:--------------------------------------------------------|:-----:|:---:|:---:|:---:|:---:|:---:|
| `AmplitudeScaling` | uniform, normal                                         |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Apodization` | exponential, truncate                                   |   ✓   | ✓† | ✓† | ✓† | ✓† | ✓† |
| `ArtificialPeaks` | Lorentzian, Gaussian, Voigt                             |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AverageSampler` | random, deterministic                                   |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `BaselineAugmentation` | random_walk, bspline, polynomial                        |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CoilSampler` | draw from the coils present, grow an array from sensitivity maps, or swap it under existing coils |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EddyCurrent` | synthetic, water                                        |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FrequencyShift` | shift_hz                                                |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Noise` | sigma, sigma_frac, snr, snr_db                          |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `KspaceUndersampling` | cartesian, gridded, nufft                               |   ~   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `LineBroadening` | lorentzian, gaussian, voigt                             |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Macromolecules` | parametrized, semi_parametrized, measured, supplied     |   ~   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `NIfTI_RawProcessor` | coil combination, alignment, averaging, ECC, phase/freq |   ✓   | ~ | ~ | ~ | ~ | ~ |
| `RawProcessor` | tensor twin of `NIfTI_RawProcessor`: same steps + HLSVD water removal |   ~   | ✓‡ | ✓‡ | ✓‡ | ✓‡ | ✓‡ |
| `PhaseShift` | zero_order, first_order                                 |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ResidualWater` | lorentzian                                              |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SpatialAugmentations` | 2-D / 3-D affine, flip, zoom, shear                     |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `SpuriousEchoes` | replica, hybrid                                         |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Tap` | identity marker: snapshot a stage for `outputs`         |   ✓   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ZeroFill` | pad FID to target length                                |   ✓   | ✓† | ✓† | ✓† | ✓† | ✓† |

**✓** native — data tensor stays in the target framework throughout.  
**~** automatic — the base class routes the call through the NIfTI-list path; the result is returned in the original backend format. Functional, but not natively differentiable.  
**—** not supported / not applicable.

Modes that differ in backend support get a row each, named `Module[mode]`.

> **†** `Apodization[truncate]` and `ZeroFill` both change `N_PTS`. Because `target_pts` / `n_pts` is a module-level scalar, the **same length is applied uniformly to every batch member** — the output is still a uniform tensor.

> **‡** `RawProcessor` matches `NIfTI_RawProcessor` to floating-point tolerance, with three deviations: outlier removal **masks** dynamics instead of dropping them (consumed exactly by averaging; zeroed and exposed as `last_keep_mask_` when not averaging), `registration_method='own'` trades exact FSL-MRS alignment for a much faster vectorized search, and water removal uses a truncated SVD. Gradients flow through the signal path only.

---

## Contact

For questions, issues, or collaborations:

- **GitHub Issues**: [github.com/julianmer/Augmentrum/issues](https://github.com/julianmer/Augmentrum/issues)
- **Email**: jlamaste@gmail.com, j.p.merkofer@tue.nl, kci2104@columbia.edu

---

## Citation
J. T. LaMaster, J. P. Merkofer, K. C. Igwe, "Augmentrum: A Data Augmentation Package for MR Spectroscopy", _International Society for Magnetic Resonance in Medicine (ISMRM)_, Abstract #05685, Cape Town, South Africa, 2026.

---

<div align="center">
  <sub>Built with ❤️ for the MRS community</sub>
</div>
