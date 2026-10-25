# Handover: using GIRF-Sim for MRI simulations (updated)

GIRF-Sim repo: `/run/media/johnlamaster/data/Postdoc/GIRF-Sim`. It is installed editable in the `python3.13` conda env, so `import girf_module, seq_girf, bacon_girf, mrs_seq` picks up changes without reinstalling.

**Status:** these changes are not yet committed in GIRF-Sim. Pull or check them out once they are.

## What changed

1. **Standard imaging sequences now exist.** Earlier, the only spatially encoded `.seq` files were PRESS/STEAM/sLASER-localized spiral MRSI. Imaging-like files built from those are not a clinical protocol and were removed. MRI simulations should use the new files:
   - **Generation:** `python -m mrs_seq.generate_all [out_dir] [--protocols {all,mrs,imaging}]` (default `all`, into `seq_output/`).
   - **`seq_output/spiral_gre_2d.seq`:** spoiled spiral GRE, single slice. 5 mm slice, 220 mm FOV, 128 × 128 matrix, 32 interleaves, 8.06 ms readout per TR. TR 15 ms, TE 1.19 ms (minimum; RF centre to the start of the spiral-out readout), flip angle 10°. 100 dummy TRs, then 32 acquiring TRs.
   - **`seq_output/spiral_gre_3d_stack.seq`:** the same protocol as a 3D stack of spirals. 64 mm slab in 32 partitions of 2 mm, 1024 acquiring TRs, TE 0.97 ms.
   - **Both files:** quadratic RF spoiling (117° increment), applied to both the RF and the ADC phase offsets. A z spoiler at the end of each TR (in 3D it also rewinds the kz encode). Peak 28.6 mT/m and 127 T/m/s, within the generic 3 T limits of 40 mT/m and 150 T/m/s.
   - **Builder:** `mrs_seq.imaging.build_spiral_gre(system, acquisition_mode="mri2d"|"mri3d", fov_mm, matrix, slice_thickness_mm, n_shots, flip_angle_deg, TE=None, TR, n_dummy, ...)`. It returns `(seq, meta)`, and `meta["readouts"]` holds each shot's target and achieved k-space. Single-pass readouts come from `mrs_seq.spectroscopy.build_mri_readout`.
2. **`seq_girf.load_all_shots_batched` (and therefore `GIRFModule.from_measured`) now skips TRs without an ADC**, i.e. the dummy scans. Without this fix the GRE files would produce 132 "shots" with the window set to the whole TR. If you have your own loader, apply the same rule.
3. **MRSI `.seq` headers are corrected.** `SpectralBandwidth` and `SpectralPoints` now report what the spiral MRSI readout actually gives (about 1099 Hz and 563 points), not the SVS values (2000 Hz, 1024 points). SVS headers are unchanged.

## Typical use

```python
import bacon_girf as bg, girf_module as gm
bacon = bg.load_bacon_girf("<GIRF-Sim>/local_data/zenodo/zenodo/GIRFs", order=1)
mod = gm.GIRFModule.from_measured("<GIRF-Sim>/seq_output/spiral_gre_2d.seq", bacon=bacon)
k_nom, k_act = mod.k_nominal, mod.k_actual   # [n_shots, 3, n_adc] cycles/m, 10 us raster
# encode with k_act, reconstruct with k_nom -> GIRF-induced image error
```

For a custom trajectory (no `.seq` file): build gradients in mT/m on a 10 µs raster, add a zero-gradient lead-in of about 1 ms so the GIRF sees the waveform start from rest, and call `bg.bacon_kspace_error_batch(bacon, g_mT[S,3,T], k_nominal[S,3,T], dt)`. Then slice off the lead-in. `scripts/ismrm_figures/fig2_measured_girf_mri.py::build_spiral_girf_trajectory` is a worked example. It also shows how to read k(t) at an ADC dwell different from the gradient raster.

## Limitations to keep in mind

- **Gradient history:** the batched loader applies the GIRF to the readout window only. Gradients before the window (excitation rephaser, kz encode) and from previous TRs (eddy and mechanical terms with long time constants, relevant at TR 15 ms) are not included. If those matter, use the full waveform with the per-shot `bacon_kspace_error_batch` path.
- **kz lost in 3D:** batched k-space is re-referenced to k = 0 at each readout start, so the kz encode of 3D files does not appear in `k_nominal`. Take kz from `meta["readouts"][i].kz_index` or `trajectory_meta` instead.
- **Density compensation:** use Augmentrum's `GriddingNUFFT.density_weights` with the fixed version (separate density-compensation kernel, physical scale; currently uncommitted in the Augmentrum repo). The old version gave 12–33 % radial weighting errors at `osf=4` and a brightness that depended on the trajectory. Details are in `<GIRF-Sim>/AUGMENTRUM_DCF_HANDOVER.md`.
- **Speed matters:** the GIRF error grows with readout speed, roughly with gradient amplitude, and faster than linearly in practice. In Figure 2, the same fully sampled 128 × 128 spiral read in 4.1 ms instead of 9.1 ms (14 against 6 mT/m) raised image NRMSE from 0.04 to 0.18, mostly as low-k contrast loss. Report gradient amplitude, slew and readout duration for every trajectory you compare.
- **Measured GIRF:** Bacon et al., order ≤ 1. The released processed data has 14.29 Hz resolution; a 5 Hz recomputation from the raw data is in `local_data/updated_GIRF/roPts40000/GIRFs`.
