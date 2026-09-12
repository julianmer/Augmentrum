# Handover: Low-Field MRI Synthesis Pipeline for Augmentrum

## 1. Objective

Implement a modular low-field MRI synthesis pipeline in Augmentrum that converts high-field MRI images, typically 3T structural images, into synthetic LF MRI acquisitions.

The pipeline must explicitly separate:

1. **Empirical LF contrast transformation**
2. **LF acquisition/system simulation**
3. **LF reconstruction**

The contrast module must not simulate acquisition noise or k-space artifacts. The acquisition module must operate on complex-valued data and be responsible for trajectory, sampling, system effects, noise and reconstruction.

The pipeline should be designed so that the current empirical histogram method can later be replaced or supplemented by quantitative T1/T2/PD-based contrast simulation.

---

## 2. High-Level Pipeline

```text
3T image
  ↓
preprocessing
  ↓
empirical LF contrast mapping
  ↓
complex object formation
  ↓
sample coherent LF scanner profile
  ↓
coil / B0 / B1 / system effects
  ↓
forward k-space operator
  ↓
undersampling / trajectory
  ↓
complex Gaussian acquisition noise
  ↓
LF reconstruction
  ↓
synthetic LF image
```

---

## 3. Important Scientific Constraints

### 3.1 Empirical contrast mapping

The histogram mapper represents **empirical LF image contrast**, not Bloch-equation physics.

Do not describe the resulting contrast transform as a physical model of T1/T2 relaxation.

Use sequence/protocol-specific target distributions.

At minimum, distinguish:

```text
sequence_family
TR
TE
TI
flip_angle
field_strength
resolution
bandwidth
reconstruction
```

Clinical labels such as `T1w`, `T2w` and `FLAIR` alone are insufficient.

---

## 4. Input preprocessing

The preprocessing module should support:

```text
brain/foreground mask
robust intensity clipping
input intensity normalization
optional bias-field correction
optional background exclusion
```

Histogram estimation and matching must operate on a foreground mask rather than the complete image whenever possible.

Do not allow background zeros/noise to dominate the transformation.

The mapper must not intentionally add image-domain noise.

---

## 5. Contrast mapping

Implement a monotonic continuous histogram mapping:

$$
I_{\\mathrm{LF}}(\\mathbf r)
=
Q_t
\\left(
F_s(I_{\\mathrm{HF}}(\\mathbf r))
\\right)
$$

where:

- $F_s$ is the source-image empirical CDF
- $Q_t$ is the target LF quantile function

Implementation requirements:

- estimate CDF robustly from masked voxels
- use configurable percentile clipping
- avoid numerical problems at CDF endpoints
- use interpolation rather than discrete histogram lookup
- preserve spatial coordinates
- preserve intensity rank
- expose mapping parameters for reproducibility

Use a configurable epsilon such as:

```text
p ∈ [epsilon, 1 - epsilon]
```

before evaluating the target quantile function.

---

## 6. Target template bank

Store target LF distributions as quantile functions rather than only histograms/CDFs.

Example representation:

```python
{
    "field_strength_T": 0.075,
    "sequence_family": "T1w",
    "TR_ms": ...,
    "TE_ms": ...,
    "TI_ms": None,
    "flip_angle_deg": ...,
    "resolution_mm": [...],
    "bandwidth_Hz": ...,
    "scanner_id": ...,
    "coil_id": ...,
    "reconstruction_id": ...,
    "quantile_p": [...],
    "quantile_values": [...],
}
```

Templates should preferentially come from:

- high-SNR
- fully sampled
- well-characterized LF reference scans

Do not require mathematically noise-free images.

Separate contrast characterization from nuisance effects such as receive sensitivity and arbitrary intensity scaling where practical.

---

## 7. Template selection

Do not independently randomize contrast, noise, coil and scanner parameters.

Introduce a coherent:

```python
LFScannerProfile
```

containing all relevant parameters.

For example:

```python
LFScannerProfile(
    contrast_template=...,
    noise_model=...,
    coil_model=...,
    b0_model=...,
    b1_model=...,
    trajectory=...,
    gnl_model=...,
    girf_model=...,
    concomitant_model=...,
    reconstruction=...,
)
```

Randomly sample a scanner profile for each synthetic acquisition.

This allows realistic correlations between contrast, coil behavior, noise and acquisition hardware.

---

## 8. Template interpolation

Support two modes.

### Mode A: template selection

Select one measured template from the compatible template bank.

### Mode B: template interpolation

Select $N$ compatible templates and Dirichlet weights

$$
w_i \geq 0,
\\qquad
\\sum_i w_i=1.
$$

Interpolate quantile functions:

$$
Q_{\\mathrm{mix}}(p)
=
\\sum_i w_iQ_i(p).
$$

Do **not** use an arithmetic average of CDFs when the intention is to generate an intermediate scanner distribution.

Restrict interpolation to compatible templates with the same:

```text
field strength class
sequence
TR/TE/TI class
resolution class
reconstruction class
```

unless deliberately configured otherwise.

---

## 9. Complex object formation

3T inputs may be magnitude-only.

Do not claim that FFT of a magnitude image recovers the original raw k-space.

Instead treat the mapped image as a synthetic LF object.

Support:

```text
phase_mode = "zero"
phase_mode = "supplied"
phase_mode = "smooth_random"
```

Default:

```text
phase_mode = "zero"
```

The object should become complex before the forward acquisition model.

---

## 10. Coil sensitivity

The forward model must support complex receive-coil sensitivity maps:

$$
x_c(\\mathbf r)
=
S_c(\\mathbf r)x(\\mathbf r).
$$

Do not encode receive-coil sensitivity permanently into the histogram templates if it can be modeled separately.

Support:

```text
single_coil
multi_coil
supplied sensitivity maps
synthetic sensitivity maps
```

For multi-coil simulation, retain individual complex channels until reconstruction.

---

## 11. B0/B1 and system effects

The LF scanner profile should optionally provide:

```text
static B0 map
dynamic B0 model
B1+ magnitude
B1+ phase
B1- magnitude
B1- phase
```

and existing Augmentrum system models should be reusable wherever applicable.

Do not duplicate existing implementations.

The LF MRI pipeline should compose these effects rather than create parallel implementations.

---

## 12. Forward acquisition model

For non-Cartesian trajectories, do not implement:

```text
Cartesian FFT → interpolate/select trajectory points
```

as the general forward model.

Instead evaluate the Fourier encoding at the actual trajectory:

$$
y_c(t)
=
\\mathcal{F}_{\\mathbf k(t)}
\\left\{
S_c(\\mathbf r)x(\\mathbf r)
\\right\}
+
n_c(t).
$$

Use the existing NUFFT or trajectory-aware Augmentrum infrastructure whenever possible.

For Cartesian acquisitions, a standard FFT implementation is acceptable.

The forward model must expose the actual trajectory coordinates and timing.

---

## 13. Acquisition noise

Thermal noise must be added to **complex acquisition data**, not as Rician image noise.

Use:

$$
n \sim \\mathcal{CN}(0,\\Psi)
$$

where $\\Psi$ may represent measured channel covariance.

Support:

```text
single-channel sigma
multi-channel covariance
independent complex Gaussian noise
measured covariance
```

Noise should be sampled in the acquisition domain.

Do not implement a “Rician k-space noise” mode.

Rician/noncentral-χ statistics should emerge naturally after reconstruction and magnitude formation.

---

## 14. Noise calibration

Prefer measured scanner-specific LF noise parameters over theoretical B0 scaling.

Support configuration using:

```text
noise_sigma
noise_covariance
receiver_bandwidth
receiver_gain
ADC scaling
```

The preferred acquisition calibration procedure is RF-off/noise-only measurement.

If measured LF noise is unavailable, allow a theoretical fallback model, but mark it explicitly as approximate.

---

## 15. Sampling and undersampling

Sampling must be represented explicitly as part of the acquisition model.

Support:

```text
Cartesian
radial
spiral
rosette
EPSI
other existing Augmentrum trajectories
```

Use actual acquisition coordinates and timing.

Separate:

```text
trajectory generation
trajectory/system distortion
sampling
reconstruction
```

Do not merge trajectory generation with noise injection.

---

## 16. Existing physics modules

Reuse existing Augmentrum modules whenever available for:

```text
B0
B1
GNL
GIRF / eddy currents
concomitant fields
trajectory errors
coil effects
k-space undersampling
regridding
```

The LF synthesis pipeline should orchestrate these modules rather than duplicate them.

---

## 17. Reconstruction

The reconstructed image should be generated using the same type of reconstruction intended to represent the LF scanner.

Support, where applicable:

```text
DCF + adjoint NUFFT
grating
iterative reconstruction
multi-coil reconstruction
user-supplied reconstruction
```

Do not apply a generic image-domain blur after reconstruction as a substitute for acquisition resolution.

Resolution should primarily emerge from the sampled k-space extent, trajectory and reconstruction.

---

## 18. Intensity scaling

Keep these concepts separate:

```text
LF tissue contrast
physical signal amplitude
receive sensitivity
receiver gain
reconstruction scaling
display normalization
```

Do not allow histogram matching to accidentally determine the absolute k-space noise level.

Provide explicit signal/noise scaling controls.

---

## 19. Recommended API

Suggested high-level API:

```python
lf_mri = LowFieldMRI(
    scanner_profile=profile,
    contrast_model=contrast_model,
    forward_model=forward_model,
    reconstruction=reconstruction,
)
```

Example:

```python
output = lf_mri(
    image=input_3t,
    metadata=input_metadata,
    rng=rng,
)
```

The pipeline should return, where requested:

```python
{
    "image": reconstructed_lf_image,
    "kspace": acquired_complex_kspace,
    "trajectory": trajectory,
    "scanner_profile": sampled_profile,
    "contrast_mapping": mapping_metadata,
}
```

Support taps/intermediate outputs using the existing Augmentrum pipeline mechanism.

---

## 20. Reproducibility

Every synthetic acquisition must be reproducible from:

```text
random seed
scanner profile ID
template IDs
template weights
contrast mapping parameters
trajectory parameters
system-effect parameters
noise seed
reconstruction parameters
```

Expose all sampled parameters through metadata.

---

## 21. Validation requirements

Before considering the implementation scientifically validated, compare synthetic and real LF scans using held-out subjects.

Evaluate separately:

### Contrast

```text
GM / WM / CSF intensity distributions
tissue-wise contrast ratios
histogram distance
Wasserstein distance
template-to-real distribution agreement
```

### Noise

```text
background noise variance
real/imaginary noise statistics
channel covariance
SNR
spatial noise variation after reconstruction
```

### Acquisition

```text
resolution / PSF
undersampling artifacts
trajectory artifacts
B0 distortion
GNL distortion
GIRF/eddy-current behavior
concomitant-field effects
```

### Overall image fidelity

```text
voxel-wise agreement where paired data exist
structural similarity
tissue-wise statistics
artifact distributions
lesion/pathology preservation
```

Do not validate only global histogram similarity. A synthetic image can have the correct histogram while having incorrect spatial or pathological information.

---

## 22. Future quantitative contrast model

The architecture must allow:

```text
ContrastModel
├── EmpiricalHistogramModel
└── QuantitativeRelaxationModel
```

The future quantitative model should support simulation from:

```text
PD
T1
T2
T2*
B0
B1+
sequence parameters
```

using an appropriate signal equation.

Do not implement this as part of the initial empirical histogram implementation unless already available.

The empirical model is the first implementation, not the final definition of LF contrast simulation.

---

## 23. Explicit non-goals

The initial implementation must **not**:

- add Gaussian/Rician noise in image space
- call image-domain Rician noise “k-space noise”
- claim that histogram matching reproduces LF relaxation physics
- independently randomize unrelated scanner parameters
- replace non-Cartesian forward modeling with Cartesian FFT plus interpolation
- bake coil sensitivity permanently into contrast templates
- use arbitrary histogram mixing as a proxy for a physical scanner
- assume all `T1w`, `T2w` or `FLAIR` acquisitions are interchangeable

---

## 24. Scientific positioning

The resulting pipeline should be described as:

> **Empirical LF contrast synthesis followed by physics-based LF acquisition and reconstruction simulation.**

It should **not** be described as a complete first-principles simulation of conversion from 3T MRI to 75 mT MRI.

The histogram stage captures the observed intensity distribution, whereas the downstream Augmentrum stage models the acquisition effects.
