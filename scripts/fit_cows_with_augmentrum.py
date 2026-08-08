####################################################################################################
#                                  fit_cows_with_augmentrum.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-03-09                                                                              #
#                                                                                                  #
# Purpose: Load COWS data with Augmentrum, process it, fit with FSL-MRS, and save fitting          #
#          parameters (concentrations + signal model params) for later use as simulation           #
#          parameter ranges.                                                                       #
#                                                                                                  #
# Pipeline:                                                                                        #
#   1. Load COWS data using Augmentrum's COWSDataModule                                            #
#   2. Process data with Augmentrum's NIfTI_RawProcessor (coil combine, align, average, etc.)      #
#   3. Fit processed spectra using FSLFittingFramework                                             #
#   4. Save concentrations, signal parameters, and computed ranges                                 #
#   5. Demonstrate loading back and conversion to simulation defs                                  #
#                                                                                                  #
# Output:                                                                                          #
#   results/<dataset>_<basis>/                                                                     #
#       fit_results.pkl       -- Full results (pickle)                                             #
#       fit_results.json      -- Summary (JSON)                                                    #
#       concentrations.npy    -- Concentration array (n_spectra x n_basis)                         #
#       uncertainties.npy     -- Uncertainty array                                                 #
#       params.npy            -- Signal parameters (gamma, sigma, eps, phi0, phi1, baseline)       #
#       parameter_ranges.json -- Min/max/mean/std per parameter (for simulation)                   #
#       individual_fits/      -- Per-spectrum fit plots and CSVs                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import numpy as np
import os
import sys
import time

# Add parent directory to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # Ensure relative paths like 'data/...' resolve correctly

from augmentrum.dataset.cows import COWSDataModule
from augmentrum.processing.nifti_raw_processor import NIfTI_RawProcessor
from augmentrum.core import NIfTI_MRS_Plus, Backend

from scripts.fsl_fitting_framework import FSLFittingFramework


#****************************#
#   configuration defaults   #
#****************************#
DEFAULT_CONFIG = {
    # Data paths
    'data_dir': 'data/openneuro_ds006812/',      # COWS data (openneuro TWIX format)
    'basis_dir': 'data/BasisSets/TE26_basis/',   # Default basis set (JSON format)

    # Output
    'save_dir': 'results/cows_fit/',

    # COWS data selection
    'location': None,       # None = all locations, or 'OCCIPITAL', 'PARIETAL', 'PFL'
    'water_sup': None,      # None = all, or 'VAPOR', 'COWS7', 'COWS12'

    # Fitting parameters
    'method': 'Newton',     # 'Newton' (fast) or 'MH' (Bayesian, slow)
    'model': 'free_shift',  # Voigt (gamma+sigma per group) + free eps per metabolite
    'ppmlim': (0.5, 4.2),
    'TE': 26e-3,            # Echo time (s) — COWS sLASER TE = 26 ms
    'TR': 2.0,              # Repetition time (s) — COWS TR = 2 s
    'metab_groups': 'individual',  # Per-basis Lorentzian linewidths
    'internal_ref': None,   # Auto-detect Cr+PCr for internal referencing

    # Processing
    'conj': False,          # Conjugate FIDs
    'coil_method': 'fsl-mrs',

    # Framework
    'include_params': True,  # Extract signal params (gamma, sigma, eps, ...)
    'basis_fmt': None,       # Auto-detect basis format
    'basis_bw': None,        # Override bandwidth (needed for .raw)
    'basis_cf': None,        # Override central frequency (needed for .raw)
}


#***************************#
#   data format detection   #
#***************************#
def _detect_data_format(data_dir):
    """
    Detect available data format, checking that files are actually readable
    (not just broken git-annex symlinks).

    Returns:
        'twix' - Siemens TWIX .dat files (openneuro, actually present)
        'mat' - MATLAB .mat structured arrays (COWS_mat style, directly in subject dirs)
        'mat_derivatives' - MATLAB .mat in openneuro derivatives/mrs_mat/
        'unknown' - Cannot determine
    """
    sub_dirs = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))
                       and not d.startswith('.')])
    if not sub_dirs:
        return 'unknown'

    # Check for TWIX (.dat) in sub-XX/mrs/sourcedata/
    for sub in sub_dirs:
        src = os.path.join(data_dir, sub, 'mrs', 'sourcedata')
        if os.path.isdir(src):
            dats = [f for f in os.listdir(src) if f.endswith('.dat')]
            # Check if at least one .dat is actually readable (not broken symlink)
            for dat in dats:
                full = os.path.join(src, dat)
                if os.path.isfile(full) and os.path.exists(os.path.realpath(full)):
                    return 'twix'
            # Directory exists but files are broken symlinks — try other formats
            break

    # Check for .mat in openneuro derivatives/mrs_mat/
    deriv_mat = os.path.join(data_dir, 'derivatives', 'mrs_mat')
    if os.path.isdir(deriv_mat):
        for sub_mat in os.listdir(deriv_mat):
            sub_mat_path = os.path.join(deriv_mat, sub_mat)
            if os.path.isdir(sub_mat_path):
                for root, dirs, files in os.walk(sub_mat_path):
                    for f in files:
                        full = os.path.join(root, f)
                        if f.endswith('.mat') and os.path.exists(os.path.realpath(full)):
                            return 'mat_derivatives'
                break  # Only check first subject

    # Check for .mat directly in subject dirs (COWS_mat legacy format)
    for sub in sub_dirs:
        sub_path = os.path.join(data_dir, sub)
        if os.path.isdir(sub_path):
            for root, dirs, files in os.walk(sub_path):
                for f in files:
                    full = os.path.join(root, f)
                    if f.endswith('.mat') and os.path.exists(os.path.realpath(full)):
                        return 'mat'
            break  # Only check first subject

    return 'unknown'


#***************************#
#   main fitting pipeline   #
#***************************#
def run_fitting_pipeline(config):
    """
    Main pipeline: Load COWS → Process → Fit → Save.

    Args:
        config: Configuration dict (see DEFAULT_CONFIG for keys)

    Returns:
        results: Fitting results dict
        ranges: Parameter ranges dict
    """
    start_time = time.time()

    print("=" * 80)
    print("  COWS Data Fitting Pipeline with Augmentrum + FSL-MRS")
    print("=" * 80)

    #***********************************************************#
    #   step 1: load cows data using augmentrum's data module   #
    #***********************************************************#
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 1: Loading COWS Data                                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    data_dir = config['data_dir']
    print(f"  Data directory: {data_dir}")
    print(f"  Location filter: {config['location'] or 'ALL'}")
    print(f"  Water suppression filter: {config['water_sup'] or 'ALL'}")

    # ── Detect data format: what is actually available on disk? ──
    # Priority: TWIX .dat (if fetched) > .mat derivatives (if fetched) > COWS_mat legacy
    data_format = _detect_data_format(data_dir)
    print(f"  Detected format: {data_format}")

    # For .mat data in openneuro derivatives, redirect to that subfolder
    effective_data_dir = data_dir
    if data_format == 'mat_derivatives':
        effective_data_dir = os.path.join(data_dir, 'derivatives', 'mrs_mat')
        print(f"  Using derivatives: {effective_data_dir}")

    loader = COWSDataModule(
        data_dir=effective_data_dir if data_format != 'twix' else data_dir,
        location=config['location'],
        water_sup=config['water_sup']
    )

    loading_failures = []

    if data_format == 'twix':
        print("  Loading format: Siemens TWIX (openneuro)")
        data, water, mm_data, mm_water, names = loader.load_twix()
        loading_failures = getattr(loader, 'load_failures', [])
    elif data_format in ('mat', 'mat_derivatives'):
        print("  Loading format: MATLAB (.mat) structured arrays")
        data, water, mm_data, mm_water, names = loader.load_mats()
        loading_failures = getattr(loader, 'load_failures', [])
    else:
        # Data not found or not fetched — give helpful error
        print(f"\n  ⚠ No readable data found in {data_dir}")
        print(f"  The openneuro COWS data uses git-annex. To fetch it:")
        print(f"    cd {data_dir} && datalad get .")
        print(f"  Or use the pre-processed COWS_mat data if available:")
        print(f"    --data data/COWS/COWS_mat/")
        raise ValueError(f"No readable data files in {data_dir}. "
                         f"Run 'datalad get .' in the dataset directory, "
                         f"or use --data data/COWS/COWS_mat/ for pre-processed data.")

    print(f"\n  Loaded: {len(data)} metabolite spectra")
    print(f"          {len(water)} water references")
    print(f"          {len(mm_data)} macromolecule spectra")
    print(f"  Names: {names[:5]}{'...' if len(names) > 5 else ''}")

    if loading_failures:
        print(f"\n  ── LOADING FAILURES ({len(loading_failures)}) ──")
        for fname, err in loading_failures:
            print(f"    ✗ {fname}")
            print(f"      Reason: {err}")

    if len(data) == 0:
        print("\n  ⚠ No data loaded! Check data_dir and filters.")
        return None, None

    #******************************************************#
    #   step 2: process data using augmentrum's pipeline   #
    #******************************************************#
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 2: Processing Data with Augmentrum                         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    processed_data = []
    processed_water = []
    processed_names = []
    processing_failures = []  # Track (name, error) for end report

    if data_format == 'twix':
        # ─── TWIX data: NIfTI-MRS objects → full Augmentrum processing pipeline ───
        print("  Data type: NIfTI-MRS objects → full processing pipeline")

        processor = NIfTI_RawProcessor(
            conj=config['conj'],
            coil=True,
            align=True,
            remove_outliers=True,
            average=True,
            ecc=True if water else False,
            truncate=False,
            remove_water=False,
            shift_ref=True,
            phase_correct=True,
            coil_method=config['coil_method'],
        )

        for i in range(len(data)):
            print(f"  Processing spectrum {i+1}/{len(data)}: {names[i]}...", end=' ')

            try:
                subj_data = data[i]
                subj_water = water[i] if i < len(water) else None

                # Wrap in NIfTI_MRS_Plus for pipeline
                data_plus = NIfTI_MRS_Plus(
                    nifti_list=[subj_data],
                    backend=Backend.NIFTI_LIST,
                    volatile=True
                )
                water_plus = NIfTI_MRS_Plus(
                    nifti_list=[subj_water],
                    backend=Backend.NIFTI_LIST,
                    volatile=True
                ) if subj_water is not None else None

                # Apply Augmentrum processing pipeline
                proc_data, proc_water = processor(data_plus, water_plus)

                processed_data.append(proc_data.list()[0])
                if proc_water is not None:
                    processed_water.append(proc_water.list()[0])
                processed_names.append(names[i])

                print("✓")
            except Exception as e:
                processing_failures.append((names[i], str(e)))
                print(f"✗ Error: {e}")
                continue

    elif data_format == 'mat':
        # The .mat files contain structured MATLAB arrays with fields:
        #   exptDat[0][0]['fid']  -> complex FID data
        #   exptDat[0][0]['sf']   -> spectrometer frequency (MHz)
        #   exptDat[0][0]['sw_h'] -> spectral bandwidth (Hz)
        #   exptDat[0][0]['nspecC'] -> number of spectral points
        print("  Data type: MATLAB structured arrays → extracting FID + metadata")

        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

        for i in range(len(data)):
            print(f"  Converting spectrum {i+1}/{len(data)}: {names[i]}...", end=' ')

            try:
                raw = data[i]

                # Extract FID and metadata from structured MATLAB array
                if hasattr(raw, 'dtype') and raw.dtype.names is not None:
                    # Structured array: exptDat[0] from load_mats()
                    rec = raw[0] if raw.shape else raw
                    fid = np.squeeze(rec['fid']).astype(complex)
                    sf = float(np.squeeze(rec['sf']))     # MHz
                    sw_h = float(np.squeeze(rec['sw_h'])) # Hz
                else:
                    # Already a plain array (shouldn't happen but handle it)
                    fid = np.squeeze(np.array(raw, dtype=complex))
                    sf = 123.26   # default 3T Siemens
                    sw_h = 4000.0 # default bandwidth

                if config['conj']:
                    fid = np.conjugate(fid)

                # Create NIfTI-MRS object with actual metadata from the .mat file
                nifti_data = gen_nifti_mrs(
                    data=fid.reshape((1, 1, 1) + fid.shape),
                    dwelltime=1.0 / sw_h,
                    spec_freq=sf,
                    nucleus='1H',
                    dim_tags=[None, None, None],
                    no_conj=True,
                )
                processed_data.append(nifti_data)

                # Water reference
                if i < len(water) and water[i] is not None:
                    raw_w = water[i]
                    if hasattr(raw_w, 'dtype') and raw_w.dtype.names is not None:
                        rec_w = raw_w[0] if raw_w.shape else raw_w
                        water_fid = np.squeeze(rec_w['fid']).astype(complex)
                        sf_w = float(np.squeeze(rec_w['sf']))
                        sw_h_w = float(np.squeeze(rec_w['sw_h']))
                    else:
                        water_fid = np.squeeze(np.array(raw_w, dtype=complex))
                        sf_w, sw_h_w = sf, sw_h

                    if config['conj']:
                        water_fid = np.conjugate(water_fid)

                    nifti_water = gen_nifti_mrs(
                        data=water_fid.reshape((1, 1, 1) + water_fid.shape),
                        dwelltime=1.0 / sw_h_w,
                        spec_freq=sf_w,
                        nucleus='1H',
                        dim_tags=[None, None, None],
                        no_conj=True,
                    )
                    processed_water.append(nifti_water)

                processed_names.append(names[i])
                print("✓")
            except Exception as e:
                processing_failures.append((names[i], str(e)))
                print(f"✗ Error: {e}")
                continue

    print(f"\n  Successfully processed: {len(processed_data)} spectra")
    if processing_failures:
        print(f"  Processing failures:   {len(processing_failures)}")

    if len(processed_data) == 0:
        print("\n  ⚠ No spectra processed successfully!")
        return None, None

    #************************************************************#
    #   step 3: fit with fsl-mrs using the fslfittingframework   #
    #************************************************************#
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 3: Fitting Spectra with FSL-MRS                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Auto-detect data bandwidth and sample points from processed data
    # These are used for basis resampling (NOT for constructing the basis itself).
    # The basis bw/cf come from the basis format headers or companion files.
    basis_bw = config['basis_bw']  # Explicit override for basis loading
    basis_cf = config['basis_cf']  # Explicit override for basis loading
    data_bw = None
    data_npts = None
    if len(processed_data) > 0:
        first_spec = processed_data[0]
        if hasattr(first_spec, 'hdr_ext'):
            try:
                data_bw = float(first_spec.bandwidth)
                data_npts = int(np.squeeze(first_spec[:]).shape[-1])
                # Only use data bw/cf for basis loading if user explicitly asked (basis_bw/cf set)
                # For .raw format, the framework will try to infer from companion .mat files
                if basis_cf is None:
                    basis_cf = float(first_spec.spectrometer_frequency[0])
                    print(f"  Auto-detected central frequency: {basis_cf:.4f} MHz")
                print(f"  Data bandwidth: {data_bw:.1f} Hz, sample points: {data_npts}")
            except Exception as e:
                print(f"  ⚠ Could not auto-detect data params: {e}")

    framework = FSLFittingFramework(
        path2basis=config['basis_dir'],
        method=config['method'],
        model=config.get('model', 'free_shift'),
        ppmlim=config['ppmlim'],
        conj=config['conj'],
        unc='perc',
        save_path=save_dir,
        bandwidth=data_bw,
        sample_points=data_npts,
        TE=config['TE'],
        TR=config['TR'],
        include_params=config['include_params'],
        basis_fmt=config['basis_fmt'],
        basis_bw=basis_bw,
        basis_cf=basis_cf,
        metab_groups=config.get('metab_groups', 'individual'),
        internal_ref=config.get('internal_ref'),
    )

    # Fit all processed spectra
    results = framework.fit_nifti_list(
        data_list=processed_data,
        water_list=processed_water if processed_water else None,
        names=processed_names
    )

    #***********************************************#
    #   step 4: save results and parameter ranges   #
    #***********************************************#
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  Step 4: Saving Results and Parameter Ranges                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    framework.save_results(results, save_dir)

    # Compute and display ranges
    ranges = framework.compute_parameter_ranges(results)

    # ── Raw amplitude ranges ──
    print("\n" + "─" * 70)
    print("  CONCENTRATION RANGES — RAW AMPLITUDES")
    print("─" * 70)
    print(f"  {'Metabolite':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("  " + "─" * 55)
    for name, info in ranges['concentrations'].items():
        print(f"  {name:<15} {info['low_limit']:>10.4f} {info['up_limit']:>10.4f} "
              f"{info['mean']:>10.4f} {info['std']:>10.4f}")

    # ── Internal referencing (/tCr) ──
    concs_int = results.get('concs_internal')
    if concs_int is not None and not np.all(np.isnan(concs_int)):
        int_ref_str = '+'.join(framework.internal_ref)
        print(f"\n" + "─" * 70)
        print(f"  CONCENTRATION RANGES — RELATIVE (/{int_ref_str})")
        print("─" * 70)
        print(f"  {'Metabolite':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        print("  " + "─" * 55)
        for j, name in enumerate(results['basis_names']):
            col = concs_int[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                print(f"  {name:<15} {np.min(valid):>10.4f} {np.max(valid):>10.4f} "
                      f"{np.mean(valid):>10.4f} {np.std(valid):>10.4f}")

    # ── Absolute quantification (molarity) ──
    concs_mol = results.get('concs_molarity')
    if concs_mol is not None and not np.all(np.isnan(concs_mol)):
        print(f"\n" + "─" * 70)
        print("  CONCENTRATION RANGES — ABSOLUTE (mM, water-referenced)")
        print("─" * 70)
        print(f"  {'Metabolite':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        print("  " + "─" * 55)
        for j, name in enumerate(results['basis_names']):
            col = concs_mol[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                print(f"  {name:<15} {np.min(valid):>10.4f} {np.max(valid):>10.4f} "
                      f"{np.mean(valid):>10.4f} {np.std(valid):>10.4f}")

    # ── Per-metabolite Lorentzian linewidths (gamma) ──
    if 'signal_params' in ranges:
        gamma_params = {k: v for k, v in ranges['signal_params'].items()
                        if k.startswith('gamma_')}
        sigma_params = {k: v for k, v in ranges['signal_params'].items()
                        if k.startswith('sigma_')}
        eps_params = {k: v for k, v in ranges['signal_params'].items()
                      if k.startswith('eps_')}

        if gamma_params:
            print(f"\n" + "─" * 70)
            print("  PER-METABOLITE LORENTZIAN LINEWIDTHS (gamma, Hz)")
            print("─" * 70)
            print(f"  {'Metabolite':<15} {'Min':>8} {'Max':>8} {'Mean':>8} {'FWHM':>10}")
            print("  " + "─" * 50)
            for pname, info in gamma_params.items():
                metab = pname.replace('gamma_', '')
                fwhm_mean = 2 * info['mean'] / np.pi
                print(f"  {metab:<15} {info['low_limit']:>8.2f} {info['up_limit']:>8.2f} "
                      f"{info['mean']:>8.2f} {fwhm_mean:>10.2f}")

        if sigma_params:
            print(f"\n" + "─" * 70)
            print("  PER-METABOLITE GAUSSIAN LINEWIDTHS (sigma, Hz)")
            print("─" * 70)
            print(f"  {'Metabolite':<15} {'Min':>8} {'Max':>8} {'Mean':>8}")
            print("  " + "─" * 40)
            for pname, info in sigma_params.items():
                metab = pname.replace('sigma_', '')
                print(f"  {metab:<15} {info['low_limit']:>8.4f} {info['up_limit']:>8.4f} "
                      f"{info['mean']:>8.4f}")

        if eps_params:
            print(f"\n" + "─" * 70)
            print("  PER-METABOLITE FREQUENCY SHIFTS (eps, Hz)")
            print("─" * 70)
            print(f"  {'Metabolite':<15} {'Min':>8} {'Max':>8} {'Mean':>8}")
            print("  " + "─" * 40)
            for pname, info in eps_params.items():
                metab = pname.replace('eps_', '')
                print(f"  {metab:<15} {info['low_limit']:>8.2f} {info['up_limit']:>8.2f} "
                      f"{info['mean']:>8.2f}")

    # ── Global phase ──
    phi0 = ranges.get('signal_params', {}).get('phi0')
    phi1 = ranges.get('signal_params', {}).get('phi1')
    if phi0 or phi1:
        print(f"\n" + "─" * 70)
        print("  GLOBAL PHASE PARAMETERS")
        print("─" * 70)
        if phi0:
            print(f"  phi0 (rad):    [{phi0['low_limit']:.4f}, {phi0['up_limit']:.4f}]"
                  f"  ({np.degrees(phi0['low_limit']):.1f}° to {np.degrees(phi0['up_limit']):.1f}°)")
        if phi1:
            print(f"  phi1 (rad/Hz): [{phi1['low_limit']:.6f}, {phi1['up_limit']:.6f}]")

    if 'lineshape' in ranges:
        print(f"\n  Lineshape model: {ranges['lineshape']['type']}")
        print(f"  Note: {ranges['lineshape']['note']}")

    #*******************************************************#
    #   end-of-pipeline report: all failures in one place   #
    #*******************************************************#
    fit_failures = results.get('failures', {})
    n_loaded = len(data)
    n_load_fail = len(loading_failures)
    n_processed = len(processed_data)
    n_proc_fail = len(processing_failures)
    n_fit_ok = sum(1 for c in results['concs_raw'] if not np.all(np.isnan(c)))
    n_fit_fail = len(fit_failures.get('fit_errors', []))
    n_quant_fail = len(fit_failures.get('quant_errors', []))
    n_save_warn = len(fit_failures.get('save_errors', []))
    n_has_molarity = sum(1 for c in results.get('concs_molarity', [])
                         if not np.all(np.isnan(c)))

    print(f"\n{'═' * 70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Expected spectra (10 subj × 3 loc × 3 WS):  90")
    print(f"  Loading failures:         {n_load_fail}")
    print(f"  Loaded from disk:         {n_loaded}")
    print(f"  Processing failures:      {n_proc_fail}")
    print(f"  Processed successfully:   {n_processed}")
    print(f"  Fit failures:             {n_fit_fail}")
    print(f"  Fitted successfully:      {n_fit_ok}")
    print(f"  Quantification failures:  {n_quant_fail}")
    print(f"  With absolute quant (mM): {n_has_molarity}")
    print(f"  With internal ref:        "
          f"{sum(1 for c in results.get('concs_internal', []) if not np.all(np.isnan(c)))}")
    if n_save_warn > 0:
        print(f"  Save warnings:            {n_save_warn}")

    any_failures = n_load_fail + n_proc_fail + n_fit_fail + n_quant_fail

    if n_load_fail > 0:
        print(f"\n  ── LOADING FAILURES ({n_load_fail}) ──")
        for fname, err in loading_failures:
            print(f"    ✗ {fname}")
            print(f"      Reason: {err}")

    if n_proc_fail > 0:
        print(f"\n  ── PROCESSING FAILURES ({n_proc_fail}) ──")
        for name, err in processing_failures:
            print(f"    ✗ {name}")
            print(f"      Reason: {err}")

    if n_fit_fail > 0:
        print(f"\n  ── FIT FAILURES ({n_fit_fail}) ──")
        for name, err in fit_failures.get('fit_errors', []):
            print(f"    ✗ {name}")
            print(f"      Reason: {err}")

    if n_quant_fail > 0:
        print(f"\n  ── QUANTIFICATION FAILURES ({n_quant_fail}) ──")
        for name, err in fit_failures.get('quant_errors', []):
            print(f"    ⚠ {name}")
            print(f"      Reason: {err}")

    if n_save_warn > 0:
        print(f"\n  ── SAVE WARNINGS ({n_save_warn}) ──")
        for name, err in fit_failures.get('save_errors', []):
            print(f"    ⚠ {name}")
            print(f"      Reason: {err}")

    if any_failures == 0:
        print(f"\n  ✓ All spectra loaded, processed, fitted, and quantified successfully!")

    elapsed = time.time() - start_time
    print(f"\n{'═' * 70}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results saved to: {save_dir}")
    print(f"{'═' * 70}")

    return results, ranges


#****************************************#
#   demonstrate loading and conversion   #
#****************************************#
def demo_load_and_convert(save_dir):
    """
    Demonstrate loading saved results and converting to simulation definitions.

    This shows how to:
    1. Load previously saved fitting results
    2. Convert parameter ranges to simulation-compatible format
    3. Use ranges for Augmentrum configuration
    """
    print("\n" + "=" * 80)
    print("  Demo: Loading Results and Converting to Simulation Definitions")
    print("=" * 80)

    # Load results
    loaded = FSLFittingFramework.load_results(save_dir)

    if 'ranges' not in loaded:
        print("  No parameter ranges found!")
        return

    ranges = loaded['ranges']

    # Convert to simulation definitions (MRS project format)
    conc_defs, signal_params = FSLFittingFramework.ranges_to_simulation_defs(
        ranges, margin=0.1
    )

    print("\n  Concentration definitions (simulationDefs format):")
    print("  " + "─" * 65)
    print(f"  {'Metabolite':<15} {'low_limit':>12} {'up_limit':>12}")
    print("  " + "─" * 45)
    for name, info in conc_defs.items():
        print(f"  {name:<15} {info['low_limit']:>12.4f} {info['up_limit']:>12.4f}")

    # Organize signal params by type
    gamma_p = {k: v for k, v in signal_params.items() if k.startswith('gamma_')}
    sigma_p = {k: v for k, v in signal_params.items() if k.startswith('sigma_')}
    eps_p   = {k: v for k, v in signal_params.items() if k.startswith('eps_')}
    global_p = {k: v for k, v in signal_params.items()
                if not k.startswith(('gamma_', 'sigma_', 'eps_', 'baseline_', 'conc_'))}
    baseline_p = {k: v for k, v in signal_params.items() if k.startswith('baseline_')}

    if gamma_p:
        print(f"\n  Per-metabolite Lorentzian gamma (Hz):")
        print("  " + "─" * 55)
        print(f"  {'Metabolite':<15} {'Min':>10} {'Max':>10} {'Mean':>10}")
        print("  " + "─" * 50)
        for pname, info in gamma_p.items():
            metab = pname.replace('gamma_', '')
            print(f"  {metab:<15} {info['low_limit']:>10.2f} {info['up_limit']:>10.2f} "
                  f"{info['mean']:>10.2f}")

    if sigma_p:
        print(f"\n  Per-metabolite Gaussian sigma (Hz):")
        print("  " + "─" * 55)
        print(f"  {'Metabolite':<15} {'Min':>10} {'Max':>10} {'Mean':>10}")
        print("  " + "─" * 50)
        for pname, info in sigma_p.items():
            metab = pname.replace('sigma_', '')
            print(f"  {metab:<15} {info['low_limit']:>10.2f} {info['up_limit']:>10.2f} "
                  f"{info['mean']:>10.2f}")

    if eps_p:
        print(f"\n  Per-metabolite frequency shift eps (Hz):")
        print("  " + "─" * 55)
        print(f"  {'Metabolite':<15} {'Min':>10} {'Max':>10} {'Mean':>10}")
        print("  " + "─" * 50)
        for pname, info in eps_p.items():
            metab = pname.replace('eps_', '')
            print(f"  {metab:<15} {info['low_limit']:>10.2f} {info['up_limit']:>10.2f} "
                  f"{info['mean']:>10.2f}")

    if global_p:
        print(f"\n  Global parameters:")
        print("  " + "─" * 55)
        for pname, info in global_p.items():
            print(f"    {pname}: [{info['low_limit']:.4f}, {info['up_limit']:.4f}] "
                  f"(mean={info['mean']:.4f})")

    if baseline_p:
        print(f"\n  Baseline coefficients:")
        print("  " + "─" * 55)
        for pname, info in baseline_p.items():
            print(f"    {pname}: [{info['low_limit']:.4f}, {info['up_limit']:.4f}] "
                  f"(mean={info['mean']:.4f})")

    # Show how to use these ranges with Augmentrum
    print("\n  Example: Using ranges with Augmentrum:")
    print("  " + "─" * 55)

    # Aggregate gamma/sigma across metabolites for global summary
    if gamma_p:
        all_gamma_min = min(v['low_limit'] for v in gamma_p.values())
        all_gamma_max = max(v['up_limit'] for v in gamma_p.values())
        all_gamma_mean = np.mean([v['mean'] for v in gamma_p.values()])
        print(f"    lb_hz=({all_gamma_min:.2f}, {all_gamma_max:.2f}),  "
              f"# Lorentzian broadening range (mean={all_gamma_mean:.2f})")
    if sigma_p:
        all_sigma_min = min(v['low_limit'] for v in sigma_p.values())
        all_sigma_max = max(v['up_limit'] for v in sigma_p.values())
        all_sigma_mean = np.mean([v['mean'] for v in sigma_p.values()])
        print(f"    gb_hz=({all_sigma_min:.2f}, {all_sigma_max:.2f}),  "
              f"# Gaussian broadening range (mean={all_sigma_mean:.2f})")
    if 'phi0' in global_p:
        phi0 = global_p['phi0']
        print(f"    phase0_deg=({np.degrees(phi0['low_limit']):.1f}, "
              f"{np.degrees(phi0['up_limit']):.1f}),  # Phase from fitting")
    if eps_p:
        all_eps_min = min(v['low_limit'] for v in eps_p.values())
        all_eps_max = max(v['up_limit'] for v in eps_p.values())
        print(f"    shift_hz=({all_eps_min:.2f}, {all_eps_max:.2f}),  "
              f"# Freq shift range across metabolites")


#**********************#
#   argument parsing   #
#**********************#
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fit COWS MRS data using Augmentrum + FSL-MRS pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit all COWS data (openneuro TWIX) with default TE26 basis
  python fit_cows_with_augmentrum.py

  # Fit COWS data with LCModel .raw basis
  python fit_cows_with_augmentrum.py \\
      --basis data/BasisSets/raw_basis_functions/ \\
      --basis-bw 2000 --basis-cf 127.8

  # Fit only OCCIPITAL with VAPOR suppression
  python fit_cows_with_augmentrum.py \\
      --location OCCIPITAL --water-sup VAPOR

  # Fit with the FID-A .mat basis set
  python fit_cows_with_augmentrum.py \\
      --basis data/BasisSets/basis_functions_metab_mm_mat/

  # Load and view previously saved results
  python fit_cows_with_augmentrum.py --load-only results/cows_fit/
        """
    )

    parser.add_argument('--data', type=str, default=DEFAULT_CONFIG['data_dir'],
                        help='Path to COWS data directory')
    parser.add_argument('--basis', type=str, default=DEFAULT_CONFIG['basis_dir'],
                        help='Path to basis set (any format)')
    parser.add_argument('--save', type=str, default=DEFAULT_CONFIG['save_dir'],
                        help='Output directory for results')
    parser.add_argument('--location', type=str, default=None,
                        help='Brain location filter (OCCIPITAL, PARIETAL, PFC)')
    parser.add_argument('--water-sup', type=str, default=None,
                        help='Water suppression filter (VAPOR, COWS7, COWS12)')
    parser.add_argument('--method', type=str, default='Newton',
                        choices=['Newton', 'MH'],
                        help='FSL-MRS fitting method')
    parser.add_argument('--ppmlim', type=float, nargs=2, default=[0.5, 4.2],
                        help='PPM range for fitting')
    parser.add_argument('--TE', type=float, default=None,
                        help='Echo time in seconds (for water scaling)')
    parser.add_argument('--TR', type=float, default=None,
                        help='Repetition time in seconds (for water scaling)')
    parser.add_argument('--conj', action='store_true',
                        help='Conjugate FIDs')
    parser.add_argument('--no-params', action='store_true',
                        help='Skip signal parameter extraction')
    parser.add_argument('--basis-fmt', type=str, default=None,
                        help='Explicit basis format (json, raw, mat, txt, basis)')
    parser.add_argument('--basis-bw', type=float, default=None,
                        help='Basis bandwidth (needed for .raw)')
    parser.add_argument('--basis-cf', type=float, default=None,
                        help='Basis central frequency (needed for .raw)')
    parser.add_argument('--load-only', type=str, default=None,
                        help='Only load and display previously saved results')

    return parser.parse_args()


#**********#
#   main   #
#**********#
if __name__ == '__main__':
    args = parse_args()

    # Load-only mode
    if args.load_only:
        demo_load_and_convert(args.load_only)
        sys.exit(0)

    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update({
        'data_dir': args.data,
        'basis_dir': args.basis,
        'save_dir': args.save,
        'location': args.location,
        'water_sup': args.water_sup,
        'method': args.method,
        'ppmlim': tuple(args.ppmlim),
        'conj': args.conj,
        'include_params': not args.no_params,
        'basis_fmt': args.basis_fmt,
        'basis_bw': args.basis_bw,
        'basis_cf': args.basis_cf,
    })
    # Only override TE/TR if explicitly passed on command line
    if args.TE is not None:
        config['TE'] = args.TE
    if args.TR is not None:
        config['TR'] = args.TR

    # Run fitting pipeline
    results, ranges = run_fitting_pipeline(config)

    # Demo loading and conversion
    if results is not None:
        demo_load_and_convert(config['save_dir'])














