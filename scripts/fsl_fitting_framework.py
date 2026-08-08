####################################################################################################
#                                     fsl_fitting_framework.py                                     #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-03-09                                                                              #
#                                                                                                  #
# Purpose: FSL-MRS fitting framework wrapper for Augmentrum. Bridges the Augmentrum data pipeline  #
#          with FSL-MRS spectral fitting to extract metabolite concentrations and signal           #
#          parameters (Lorentzian lineshape per basis function). Supports loading any basis set    #
#          format via the MRS loadBasis infrastructure.                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import numpy as np
import os
import pickle

from datetime import datetime
from pathlib import Path

from fsl_mrs.core import MRS
from fsl_mrs.core.basis import Basis
from fsl_mrs.utils import fitting, mrs_io, plotting, quantify

from scipy.io import loadmat


#*****************************#
#   basis loading utilities   #
#*****************************#
def read_LCModel_raw(filename, conjugate=True):
    """
    Read LCModel (.RAW, .raw, and .H2O) file format.

    @param filename -- Path to .RAW/.H2O file.
    @param bool conjugate -- Apply conjugation upon read.

    @returns -- The basis set data/FID and header if possible.
    """
    header = []
    data = []
    in_header = False
    after_header = False
    with open(filename, 'r') as f:
        for line in f:
            if (line.find('$') > 0):
                in_header = True
            if in_header:
                header.append(line)
            elif after_header:
                data.append(list(map(float, line.split())))
            if line.find('$END') > 0:
                in_header = False
                after_header = True

    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j * data[1::2]).astype(complex)

    if conjugate:
        data = np.conj(data)

    return data, header


def _infer_bw_cf_from_companion_mat(raw_basis_path, names=None):
    """
    Try to find bandwidth and central frequency from companion .mat basis files.
    Looks in sibling directories (e.g., basis_functions_metab_mm_mat/) for matching
    INSPECTOR-format .mat files that contain sw_h and sf metadata.

    @param raw_basis_path -- Path to the .raw basis directory or file.
    @param names -- Optional list of basis names to match against.
    @returns -- (bw, cf) tuple, either or both may be None if not found.
    """
    bw, cf = None, None
    if os.path.isdir(raw_basis_path):
        parent = os.path.dirname(os.path.abspath(raw_basis_path))
    else:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(raw_basis_path)))

    # Search sibling directories for .mat basis files
    for sibling in os.listdir(parent):
        sibling_path = os.path.join(parent, sibling)
        if not os.path.isdir(sibling_path):
            continue
        mat_files = [f for f in os.listdir(sibling_path) if f.endswith('.mat')]
        if not mat_files:
            continue
        # Try to read the first .mat file
        try:
            data = loadmat(os.path.join(sibling_path, mat_files[0]))
            if 'exptDat' in data:
                rec = data['exptDat'][0][0]
                bw = float(np.squeeze(rec['sw_h']))
                cf = float(np.squeeze(rec['sf']))
                break
        except Exception:
            continue

    return bw, cf


def load_basis_any_format(path2basis, bw=None, cf=None, fmt=None):
    """
    Universal basis set loader supporting all common MRS basis formats.
    Adapted from MRS project loadBasis.py.

    Supported formats:
        - .json (FSL-MRS native)
        - .raw  (LCModel)
        - .basis (LCModel combined)
        - .mat  (FID-A / INSPECTOR)
        - .txt  (JMRUI)

    @param path2basis -- Path to basis set file or directory.
    @param bw -- Bandwidth (required for .raw format).
    @param cf -- Central frequency (required for .raw format).
    @param fmt -- Explicit format override. If None, auto-detected from extension.

    @returns -- FSL-MRS Basis object.
    """
    if fmt is None:
        if os.path.isdir(path2basis):
            files = [f for f in os.listdir(path2basis) if not f.startswith('.')]
            if files:
                fmt = files[0].split('.')[-1].lower()
            else:
                raise ValueError(f"Empty directory: {path2basis}")
        else:
            fmt = path2basis.split('.')[-1].lower()

    # JSON (FSL-MRS native format)
    if fmt == 'json':
        basis, names, headers = mrs_io.fsl_io.readFSLBasisFiles(path2basis)
        # Strip file extensions so names match FSL-MRS defaults (e.g. 'NAA' not 'NAA.json')
        names = [n.rsplit('.', 1)[0] if '.' in n else n for n in names]
        return Basis(basis, names, headers)

    # JMRUI (.txt)
    elif fmt == 'txt':
        return mrs_io.read_basis(path2basis)

    # LCModel .raw files
    elif fmt == 'raw':
        if os.path.isdir(path2basis):
            files = sorted(Path(path2basis).glob('*.raw'))
        else:
            files = [Path(path2basis)]

        basis_list = []
        names = []
        for file in files:
            data, header = read_LCModel_raw(str(file))
            name = file.stem
            names.append(name)
            basis_list.append(data)

        basis = np.asarray(basis_list).astype(complex).T
        # Keep names clean (no extension) for FSL-MRS metabolite matching
        # names already are stems from file.stem above

        # .raw files don't store bw/cf. Try to infer from companion .mat files
        # (same basis set in INSPECTOR format), then fall back to user-supplied values.
        if bw is None or cf is None:
            raw_bw, raw_cf = _infer_bw_cf_from_companion_mat(path2basis, names)
            if raw_bw is not None and bw is None:
                bw = raw_bw
                print(f"  .raw basis: inferred bw={bw} Hz from companion .mat files")
            if raw_cf is not None and cf is None:
                cf = raw_cf
                print(f"  .raw basis: inferred cf={cf} MHz from companion .mat files")

        assert bw is not None and cf is not None, \
            "Bandwidth (bw) and central frequency (cf) are required for .raw format. " \
            "Pass --basis-bw and --basis-cf, or place companion .mat basis files nearby."

        header = {'centralFrequency': cf, 'bandwidth': bw,
                  'dwelltime': 1 / bw, 'fwhm': None}
        headers = [header for _ in names]
        return Basis(basis, names, headers)

    # LCModel .basis (combined file)
    elif fmt == 'basis':
        basis, names, headers = mrs_io.lcm_io.readLCModelBasis(path2basis)
        for h in headers:
            h['fwhm'] = None
        return Basis(basis, names, headers)

    # MATLAB .mat (FID-A format)
    elif fmt == 'mat':
        if os.path.isdir(path2basis):
            mat_files = sorted([f for f in os.listdir(path2basis) if f.endswith('.mat')])
        else:
            mat_files = [os.path.basename(path2basis)]
            path2basis = os.path.dirname(path2basis)

        basis_list = []
        names = []
        headers = []
        for name in mat_files:
            filepath = os.path.join(path2basis, name)
            data = loadmat(filepath)

            # Try FID-A format first
            if 'fids' in data:
                header = {
                    'centralFrequency': float(data['txfrq']),
                    'bandwidth': int(data['spectralwidth']),
                    'dwelltime': float(data['dwelltime']),
                    'fwhm': None
                }
                basis_list.append(data['fids'])
            # Then try INSPECTOR format
            elif 'exptDat' in data:
                header = {
                    'centralFrequency': data['exptDat']['sf'][0][0],
                    'bandwidth': data['exptDat']['sw_h'][0][0],
                    'dwelltime': 1 / data['exptDat']['sw_h'][0][0],
                    'fwhm': None
                }
                basis_list.append(data['exptDat']['fid'][0][0])
            else:
                print(f"Warning: Unrecognized .mat format for {name}, skipping.")
                continue

            names.append(name)
            headers.append(header)

        # Strip extensions for clean metabolite names
        names = [n.rsplit('.', 1)[0] if '.' in n else n for n in names]
        return Basis(np.squeeze(np.array(basis_list)).T, names, headers)

    else:
        raise ValueError(f"Unsupported basis format: '{fmt}'. "
                         f"Supported: json, txt, raw, basis, mat")


#**************************************************************************************************#
#                                    Class FSLFittingFramework                                     #
#**************************************************************************************************#
#                                                                                                  #
# FSL-MRS fitting wrapper for use with Augmentrum.                                                 #
#                                                                                                  #
#**************************************************************************************************#
class FSLFittingFramework:
    """
    FSL-MRS fitting wrapper for use with Augmentrum.

    Fits MRS spectra using FSL-MRS and extracts:
      - Metabolite concentrations (or amplitudes)
      - Signal parameters: gamma (Lorentzian linewidth), sigma (Gaussian linewidth),
        eps (frequency shift), phi0 (zero-order phase), phi1 (first-order phase),
        and baseline polynomial coefficients.

    Supports:
      - Any basis set format via load_basis_any_format()
      - Batch fitting (with optional multiprocessing)
      - Water-referenced quantification
      - Saving/loading of fit results and parameter ranges
      - Lorentzian lineshape extraction per basis function

    Usage:
        >>> framework = FSLFittingFramework(
        ...     path2basis='data/BasisSets/TE26_basis/',
        ...     method='Newton',
        ...     include_params=True,
        ...     save_path='results/cows_fit/'
        ... )
        >>> results = framework.fit_nifti_list(data_list, water_list)
        >>> framework.save_results(results, 'results/cows_fit/')
    """

    # Parameter names for the signal model (FSL-MRS Voigt model)
    SIGNAL_PARAM_NAMES = [
        'gamma',      # Lorentzian broadening (Hz) - per group
        'sigma',      # Gaussian broadening (Hz) - per group
        'eps',        # Frequency shift (Hz) - per group
        'phi0',       # Zero-order phase (rad)
        'phi1',       # First-order phase (rad/Hz)
        # Baseline polynomial coefficients follow (2*(order+1) values, real+imag)
    ]

    def __init__(self, path2basis, method='Newton', multiprocessing_enabled=False,
                 ppmlim=(0.5, 4.2), conj=False, unc='perc', save_path='',
                 bandwidth=None, sample_points=None, TE=None, TR=None,
                 nucleus='1H', include_params=True, baseline_order=2,
                 basis_fmt=None, basis_bw=None, basis_cf=None,
                 model='free_shift', metab_groups='individual',
                 internal_ref=None, **kwargs):
        """
        Initialize the FSL fitting framework.

        Args:
            path2basis: Path to basis set (any format: .json, .raw, .mat, .txt, .basis)
            method: Fitting method ('Newton' or 'MH')
            ppmlim: PPM range for fitting (default: (0.5, 4.2))
            conj: Whether to conjugate FIDs (default: False)
            unc: Uncertainty type ('perc' for %SD, 'crlb' for raw CRLB)
            save_path: Path to save individual fit results (empty = don't save)
            bandwidth: Data bandwidth for basis set resampling (Hz)
            sample_points: Data sample points for basis set resampling
            TE: Echo time in seconds (needed for absolute water scaling)
            TR: Repetition time in seconds (needed for absolute water scaling)
            nucleus: Nucleus type (default: '1H')
            include_params: If True, extract signal parameters (gamma, sigma, eps, ...)
            baseline_order: Polynomial baseline order (default: 2)
            basis_fmt: Explicit basis format (None = auto-detect)
            basis_bw: Bandwidth for basis loading (needed for .raw format)
            basis_cf: Central frequency for basis loading (needed for .raw format)
            model: FSL-MRS fitting model ('voigt', 'lorentzian', 'free_shift', etc.)
            metab_groups: Metabolite grouping strategy:
                'individual' - each basis function gets its own group (per-metabolite linewidths)
                'single'     - all share one group (one linewidth for all)
                list[int]    - custom group assignment per basis function
            internal_ref: Internal reference metabolites (default: auto-detect Cr+PCr)
        """
        self.method = method
        self.multiprocessing_enabled = multiprocessing_enabled
        self.ppmlim = ppmlim
        self.conj = conj
        self.unc = unc
        self.save_path = save_path
        self.TE = TE
        self.TR = TR
        self.nucleus = nucleus
        self.include_params = include_params
        self.baseline_order = baseline_order
        self.model = model
        self._metab_groups_strategy = metab_groups
        self._internal_ref = internal_ref

        # Load basis set using universal loader
        self.basis = load_basis_any_format(
            path2basis, bw=basis_bw, cf=basis_cf, fmt=basis_fmt
        )

        # Optionally resample basis to match data
        if bandwidth is not None and sample_points is not None:
            print(f"  Resampling basis to match data: bw={bandwidth} Hz, npts={sample_points}")

            basis_npts = self.basis._raw_fids.shape[0]
            basis_dt = float(self.basis._dt)
            basis_time = float(basis_npts * basis_dt)
            required_time = float(sample_points / bandwidth)

            if basis_time < required_time * 1.01:
                pad_npts = int(np.ceil((required_time * 1.1) / basis_dt)) - basis_npts
                if pad_npts > 0:
                    pad = np.zeros((pad_npts, self.basis._raw_fids.shape[1]), dtype=complex)
                    self.basis._raw_fids = np.vstack([self.basis._raw_fids, pad])
                    print(f"  Zero-padded basis: {basis_npts} → {basis_npts + pad_npts} pts "
                          f"({basis_time:.4f}s → {(basis_npts + pad_npts) * basis_dt:.4f}s)")

            self.basis._raw_fids = self.basis.get_formatted_basis(bandwidth, sample_points)
            self.basis._dt = 1. / bandwidth
            self._data_bw = bandwidth
        else:
            self._data_bw = None

        # Store basis info
        self.basis_names = [n.split('.')[0] for n in self.basis.names]
        self.n_metabs = self.basis.n_metabs
        self.n_basis = len(self.basis.names)

        # Build metabolite groups for per-metabolite linewidths
        self.metab_groups = self._build_metab_groups()
        self.n_groups = max(self.metab_groups) + 1

        # Determine internal reference metabolites
        self.internal_ref = self._determine_internal_ref()

        print(f"FSLFittingFramework initialized:")
        print(f"  Basis: {len(self.basis.names)} functions ({self.n_metabs} metabolites)")
        print(f"  Names: {self.basis_names}")
        print(f"  Model: {self.model}")
        print(f"  Method: {self.method}")
        print(f"  PPM range: {self.ppmlim}")
        print(f"  Metab groups: {self.n_groups} groups "
              f"({'individual' if self.n_groups == self.n_basis else 'shared'})")
        print(f"  Internal ref: {self.internal_ref}")
        print(f"  Water scaling: {'Yes (TE=' + str(self.TE) + ', TR=' + str(self.TR) + ')' if self.TE and self.TR else 'No'}")
        print(f"  Include params: {self.include_params}")

    def _build_metab_groups(self):
        """Build metabolite group assignments for linewidth fitting."""
        strategy = self._metab_groups_strategy
        if isinstance(strategy, list):
            assert len(strategy) == self.n_basis
            return strategy
        elif strategy == 'individual':
            return list(range(self.n_basis))
        elif strategy == 'single':
            return [0] * self.n_basis
        else:
            raise ValueError(f"Invalid metab_groups: {strategy}")

    def _determine_internal_ref(self):
        """Auto-detect internal reference metabolites from basis names."""
        if self._internal_ref is not None:
            return self._internal_ref
        names_lower = {n.lower(): n for n in self.basis_names}
        cr = [n for n in self.basis_names if n.lower().startswith('cr') and 'no' not in n.lower()]
        pcr = [n for n in self.basis_names if n.lower().startswith('pcr') and 'no' not in n.lower()]
        if cr and pcr:
            return [cr[0], pcr[0]]
        elif cr:
            return cr[:1]
        elif 'naa' in names_lower:
            return [names_lower['naa']]
        return [self.basis_names[0]]

    def _find_water_ref_metab(self):
        """
        Find water reference metabolite info for QuantificationInfo.

        FSL-MRS expects one of: ['Cr', 'PCr'], 'Cr', 'PCr', 'NAA'.
        This basis uses non-standard names like 'Cr391', 'PCr393', etc.
        We explicitly select the best match and return the required params.

        Returns (metab, n_protons, limits) or (None, None, None) if no match.
        """
        # Standard FSL-MRS water scaling candidates
        # Preference order: Cr+PCr, Cr, PCr, NAA
        candidates = [
            (['Cr', 'PCr'],    5, (2, 5)),
            ('Cr',             5, (2, 5)),
            ('PCr',            5, (2, 5)),
            ('NAA',            3, (1.8, 2.2)),
        ]
        for metab, protons, limits in candidates:
            if isinstance(metab, list):
                if all(m in self.basis_names for m in metab):
                    return metab, protons, limits
            else:
                if metab in self.basis_names:
                    return metab, protons, limits

        # None of the standard names found — try partial matching
        # NAA is most reliable and commonly present
        naa_match = [n for n in self.basis_names if n.upper() == 'NAA']
        if naa_match:
            return naa_match[0], 3, (1.8, 2.2)

        return None, None, None

    def __call__(self, *args, **kwargs):
        """Callable interface for fitting."""
        return self.fit_nifti_list(*args, **kwargs)

    #*******************************#
    #   fit a list of NIfTI files   #
    #*******************************#
    def fit_nifti_list(self, data_list, water_list=None, names=None):
        """
        Fit a list of NIfTI-MRS objects (from Augmentrum's data pipeline).

        Returns dict with concs_raw, concs_internal, concs_molarity, concs_molality,
        uncertainties, params, param_names, basis_names, fit_objects, names.
        """
        n_spectra = len(data_list)
        print(f"\nFitting {n_spectra} spectra with FSL-MRS "
              f"(model={self.model}, method={self.method}, groups={self.n_groups})...")

        all_concs_raw = []
        all_concs_internal = []
        all_concs_molarity = []
        all_concs_molality = []
        all_uncs = []
        all_params = []
        all_param_names = []
        all_fit_objects = []
        spectrum_names = names if names else [f"spectrum_{i:04d}" for i in range(n_spectra)]

        # Failure tracking for end-of-run report
        failures = {
            'fit_errors': [],       # (name, error_msg)
            'quant_errors': [],     # (name, error_msg)
            'save_errors': [],      # (name, error_msg)
        }

        # Pre-compute water reference metabolite info
        water_ref_metab, water_ref_protons, water_ref_limits = self._find_water_ref_metab()

        for i, nifti_data in enumerate(data_list):
            print(f"  Fitting spectrum {i+1}/{n_spectra}: {spectrum_names[i]}...", end=' ')

            try:
                # Extract FID from NIfTI-MRS object or raw array
                if hasattr(nifti_data, '__getitem__') and hasattr(nifti_data, 'hdr_ext'):
                    fid = np.squeeze(nifti_data[:])
                elif isinstance(nifti_data, np.ndarray):
                    fid = np.squeeze(nifti_data)
                else:
                    fid = np.squeeze(np.array(nifti_data[:], dtype=complex))
                if self.conj:
                    fid = np.conjugate(fid)

                # Get water reference if available
                water_ref = None
                if water_list is not None and i < len(water_list):
                    water_item = water_list[i]
                    if hasattr(water_item, 'hdr_ext'):
                        water_ref = np.squeeze(water_item[:])
                    elif isinstance(water_item, np.ndarray):
                        water_ref = np.squeeze(water_item)
                    else:
                        water_ref = np.squeeze(np.array(water_item[:], dtype=complex))
                    if self.conj:
                        water_ref = np.conjugate(water_ref)

                # Create FSL-MRS MRS object
                bw_for_mrs = self._data_bw if self._data_bw is not None else self.basis.original_bw
                mrs_obj = MRS(
                    FID=fid,
                    basis=self.basis,
                    H2O=water_ref,
                    cf=self.basis.cf,
                    bw=bw_for_mrs,
                    nucleus=self.nucleus
                )
                mrs_obj.processForFitting()

                # ── Fit using specified model and per-metabolite groups ──
                fit_result = fitting.fit_FSLModel(
                    mrs_obj,
                    method=self.method,
                    ppmlim=self.ppmlim,
                    model=self.model,
                    metab_groups=self.metab_groups,
                )

                # ── Quantification (matching MRS frameworkFSL.py) ──
                # 1. Raw amplitudes (always available)
                concs_raw = np.array([fit_result.getConc()[fit_result.metabs.index(m)]
                                      for m in self.basis.names])

                # 2. Internal + absolute quantification
                concs_internal = np.full(self.n_basis, np.nan)
                concs_molarity = np.full(self.n_basis, np.nan)
                concs_molality = np.full(self.n_basis, np.nan)

                try:
                    q_info = None
                    if water_ref is not None and self.TE is not None and self.TR is not None:
                        q_kwargs = {}
                        if water_ref_metab is not None:
                            q_kwargs['water_ref_metab'] = water_ref_metab
                            q_kwargs['water_ref_metab_protons'] = water_ref_protons
                            q_kwargs['water_ref_metab_limits'] = water_ref_limits
                        q_info = quantify.QuantificationInfo(
                            self.TE, self.TR, mrs_obj.names,
                            mrs_obj.centralFrequency / 1E6,
                            **q_kwargs,
                        )

                    fit_result.calculateConcScaling(
                        mrs_obj,
                        quant_info=q_info,
                        internal_reference=self.internal_ref,
                    )

                    # Internal referencing (/tCr or similar) — always computed
                    if fit_result.concScalings.get('internal') is not None:
                        internal_scaled = fit_result.getConc(scaling='internal')
                        concs_internal = np.array([
                            internal_scaled[fit_result.metabs.index(m)]
                            for m in self.basis.names
                        ])

                    # Absolute: molarity (mM)
                    if fit_result.concScalings.get('molarity') is not None:
                        mol_scaled = fit_result.getConc(scaling='molarity')
                        concs_molarity = np.array([
                            mol_scaled[fit_result.metabs.index(m)]
                            for m in self.basis.names
                        ])

                    # Absolute: molality (mmol/kg)
                    if fit_result.concScalings.get('molality') is not None:
                        molal_scaled = fit_result.getConc(scaling='molality')
                        concs_molality = np.array([
                            molal_scaled[fit_result.metabs.index(m)]
                            for m in self.basis.names
                        ])

                except Exception as e:
                    failures['quant_errors'].append((spectrum_names[i], str(e)))
                    print(f"(quant: {e}) ", end='')

                # ── Uncertainties ──
                uncs_raw = fit_result.crlb[:self.n_basis]
                if self.unc == 'perc':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        uncs = np.sqrt(uncs_raw) / np.abs(concs_raw) * 100
                        uncs[uncs > 999] = 999
                        uncs[np.isnan(uncs)] = 999
                else:
                    uncs = uncs_raw

                all_concs_raw.append(concs_raw)
                all_concs_internal.append(concs_internal)
                all_concs_molarity.append(concs_molarity)
                all_concs_molality.append(concs_molality)
                all_uncs.append(uncs)
                all_fit_objects.append(fit_result)

                # ── Extract signal parameters (per-metabolite linewidths) ──
                if self.include_params:
                    signal_params, param_names = self._extract_signal_params(fit_result)
                    all_params.append(signal_params)
                    if not all_param_names:
                        all_param_names = param_names

                # ── Save individual result ──
                if self.save_path:
                    self._save_individual_fit(fit_result, mrs_obj, i, spectrum_names[i], failures)

                print("✓")

            except Exception as e:
                failures['fit_errors'].append((spectrum_names[i], str(e)))
                print(f"✗ Error: {e}")
                all_concs_raw.append(np.full(self.n_basis, np.nan))
                all_concs_internal.append(np.full(self.n_basis, np.nan))
                all_concs_molarity.append(np.full(self.n_basis, np.nan))
                all_concs_molality.append(np.full(self.n_basis, np.nan))
                all_uncs.append(np.full(self.n_basis, np.nan))
                all_fit_objects.append(None)
                if self.include_params:
                    # Compute expected param count based on model
                    if self.model == 'voigt':
                        n_signal = self.n_groups * 3  # gamma(G) + sigma(G) + eps(G)
                    elif self.model == 'free_shift':
                        n_signal = self.n_groups * 2 + self.n_basis  # gamma(G) + sigma(G) + eps(N)
                    elif self.model == 'free_shift_lorentzian':
                        n_signal = self.n_groups + self.n_basis  # gamma(G) + eps(N)
                    elif self.model == 'lorentzian':
                        n_signal = self.n_groups * 2  # gamma(G) + eps(G)
                    else:
                        n_signal = self.n_groups * 3  # safe fallback
                    n_total = self.n_basis + n_signal + 2 + 2 * (self.baseline_order + 1)
                    all_params.append(np.full(n_total, np.nan))

        results = {
            'concs_raw': np.array(all_concs_raw),
            'concs_internal': np.array(all_concs_internal),
            'concs_molarity': np.array(all_concs_molarity),
            'concs_molality': np.array(all_concs_molality),
            'concentrations': np.array(all_concs_raw),  # backward compat
            'uncertainties': np.array(all_uncs),
            'basis_names': self.basis_names,
            'names': spectrum_names,
            'fit_objects': all_fit_objects,
            'method': self.method,
            'model': self.model,
            'ppmlim': self.ppmlim,
            'n_metabs': self.n_metabs,
            'n_basis': self.n_basis,
            'n_groups': self.n_groups,
            'metab_groups': self.metab_groups,
            'internal_ref': self.internal_ref,
        }

        if self.include_params:
            results['params'] = np.array(all_params)
            results['param_names'] = all_param_names

        n_successful = sum(1 for c in all_concs_raw if not np.all(np.isnan(c)))
        print(f"\nFitting complete! {n_successful} "
              f"of {n_spectra} spectra fitted successfully.")

        # Store failure info in results
        results['failures'] = failures

        # ── Print failure report ──
        self._print_failure_report(n_spectra, n_successful, failures,
                                   all_concs_molarity, spectrum_names)

        return results

    def _print_failure_report(self, n_spectra, n_successful, failures,
                              concs_molarity, spectrum_names):
        """Print a comprehensive failure/success report at the end of fitting."""
        n_fit_fail = len(failures['fit_errors'])
        n_quant_fail = len(failures['quant_errors'])
        n_has_molarity = sum(1 for c in concs_molarity if not np.all(np.isnan(c)))

        print("\n" + "═" * 70)
        print("  FITTING REPORT")
        print("═" * 70)
        print(f"  Total spectra:         {n_spectra}")
        print(f"  Successfully fitted:   {n_successful}")
        print(f"  Fit failures:          {n_fit_fail}")
        print(f"  Quant failures:        {n_quant_fail}")
        print(f"  With abs. quant (mM):  {n_has_molarity}")
        print(f"  With water ref:        "
              f"{'Yes' if self.TE and self.TR else 'No (TE/TR not set)'}")

        if n_fit_fail > 0:
            print(f"\n  ── FIT FAILURES ({n_fit_fail}) ──")
            for name, err in failures['fit_errors']:
                print(f"    ✗ {name}: {err}")

        if n_quant_fail > 0:
            print(f"\n  ── QUANTIFICATION FAILURES ({n_quant_fail}) ──")
            for name, err in failures['quant_errors']:
                print(f"    ⚠ {name}: {err}")

        if failures.get('save_errors'):
            print(f"\n  ── SAVE WARNINGS ({len(failures['save_errors'])}) ──")
            for name, err in failures['save_errors']:
                print(f"    ⚠ {name}: {err}")

        print("═" * 70)


    #*******************************#
    #   extract signal parameters   #
    #*******************************#
    def _extract_signal_params(self, fit_result):
        """
        Extract ALL signal parameters from an FSL-MRS fit result.

        Parameter layouts by model (N=n_basis, G=n_groups):

        voigt:                   [conc(N)] [gamma(G)] [sigma(G)] [eps(G)] [phi0] [phi1] [baseline]
        lorentzian:              [conc(N)] [gamma(G)] [eps(G)]            [phi0] [phi1] [baseline]
        free_shift:              [conc(N)] [gamma(G)] [sigma(G)] [eps(N)] [phi0] [phi1] [baseline]
        free_shift_lorentzian:   [conc(N)] [gamma(G)] [eps(N)]            [phi0] [phi1] [baseline]

        Returns (params_array, param_names_list)
        """
        all_params = fit_result.params
        n_basis = self.n_basis
        n_groups = self.n_groups

        param_names = [f"conc_{name}" for name in self.basis_names]

        if self.model == 'voigt':
            # gamma(G), sigma(G), eps(G) — all per group
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"gamma_{name}")
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"sigma_{name}")
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"eps_{name}")

        elif self.model == 'free_shift':
            # gamma(G), sigma(G) per group; eps(N) per metabolite (free)
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"gamma_{name}")
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"sigma_{name}")
            for j in range(n_basis):
                param_names.append(f"eps_{self.basis_names[j]}")

        elif self.model == 'free_shift_lorentzian':
            # gamma(G) per group; eps(N) per metabolite (free); no sigma
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"gamma_{name}")
            for j in range(n_basis):
                param_names.append(f"eps_{self.basis_names[j]}")

        elif self.model == 'lorentzian':
            # gamma(G), eps(G) — both per group
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"gamma_{name}")
            for g in range(n_groups):
                name = self.basis_names[g] if n_groups == n_basis else f"g{g}"
                param_names.append(f"eps_{name}")

        else:
            # Unknown model — label generically
            for j in range(len(all_params) - n_basis - 2):
                param_names.append(f"model_param_{j}")

        # phi0, phi1 (global)
        param_names.extend(['phi0', 'phi1'])

        # Remaining = baseline coefficients
        n_named = len(param_names)
        n_total = len(all_params)
        for b in range(n_total - n_named):
            param_names.append(f"baseline_{b}")

        return all_params, param_names

    #********************************#
    #   save individual fit result   #
    #********************************#
    def _save_individual_fit(self, fit_result, mrs_obj, idx, name, failures=None):
        """Save individual fit result to disk. Tracks errors in failures dict."""
        save_dir = os.path.join(self.save_path, 'individual_fits')
        os.makedirs(save_dir, exist_ok=True)

        errors = []
        # Summary CSV
        try:
            fit_result.to_file(os.path.join(save_dir, f'{name}_summary.csv'), what='summary')
        except Exception as e:
            errors.append(f"summary: {e}")
        # Concentrations CSV — can fail with non-standard ref names
        try:
            fit_result.to_file(os.path.join(save_dir, f'{name}_concs.csv'), what='concentrations')
        except Exception as e:
            errors.append(f"concs: {e}")
        # Fit plot
        try:
            fit_result.plot(mrs_obj, out=os.path.join(save_dir, f'{name}_fit.png'))
        except Exception as e:
            errors.append(f"plot: {e}")
        # Interactive HTML
        try:
            plotting.plotly_fit(mrs_obj, fit_result).write_html(
                os.path.join(save_dir, f'{name}_residuals.html'))
        except Exception as e:
            errors.append(f"html: {e}")
        # Pickle
        try:
            with open(os.path.join(save_dir, f'{name}_opt.pkl'), 'wb') as f:
                pickle.dump(fit_result, f)
        except Exception as e:
            errors.append(f"pkl: {e}")

        if errors:
            msg = '; '.join(errors)
            print(f"(save: {msg}) ", end='')
            if failures is not None:
                failures['save_errors'].append((name, msg))

    #********************************#
    #   save all results to folder   #
    #********************************#
    def save_results(self, results, save_path):
        """
        Save complete fitting results to a structured folder.

        Structure:
            save_path/
                fit_results.pkl       -- Full results dict (pickle)
                fit_results.json      -- Summary (JSON, human-readable)
                concentrations.npy    -- Concentrations array
                uncertainties.npy     -- Uncertainties array
                params.npy            -- Signal parameters (if include_params)
                parameter_ranges.json -- Min/max ranges per parameter
                individual_fits/      -- Per-spectrum CSV, PNG, PKL files

        Args:
            results: Results dict from fit_nifti_list()
            save_path: Output directory
        """
        os.makedirs(save_path, exist_ok=True)

        # Save arrays
        np.save(os.path.join(save_path, 'concentrations.npy'), results['concentrations'])
        np.save(os.path.join(save_path, 'uncertainties.npy'), results['uncertainties'])
        if 'concs_internal' in results:
            np.save(os.path.join(save_path, 'concs_internal.npy'), results['concs_internal'])
        if 'concs_molarity' in results:
            np.save(os.path.join(save_path, 'concs_molarity.npy'), results['concs_molarity'])
        if 'concs_molality' in results:
            np.save(os.path.join(save_path, 'concs_molality.npy'), results['concs_molality'])
        if 'params' in results:
            np.save(os.path.join(save_path, 'params.npy'), results['params'])

        # Save pickle (full results minus fit_objects which may not pickle cleanly)
        results_for_pickle = {k: v for k, v in results.items() if k != 'fit_objects'}
        with open(os.path.join(save_path, 'fit_results.pkl'), 'wb') as f:
            pickle.dump(results_for_pickle, f)

        # Save fit objects separately
        fit_objects = results.get('fit_objects', [])
        for i, obj in enumerate(fit_objects):
            if obj is not None:
                try:
                    with open(os.path.join(save_path, f'fit_obj_{i:04d}.pkl'), 'wb') as f:
                        pickle.dump(obj, f)
                except:
                    pass

        # Save JSON summary
        summary = self._build_json_summary(results)
        with open(os.path.join(save_path, 'fit_results.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save parameter ranges
        ranges = self.compute_parameter_ranges(results)
        with open(os.path.join(save_path, 'parameter_ranges.json'), 'w') as f:
            json.dump(ranges, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {save_path}")
        print(f"  - fit_results.pkl (full results)")
        print(f"  - fit_results.json (summary)")
        print(f"  - concentrations.npy ({results['concentrations'].shape})")
        print(f"  - uncertainties.npy ({results['uncertainties'].shape})")
        if 'params' in results:
            print(f"  - params.npy ({results['params'].shape})")
        print(f"  - parameter_ranges.json (min/max per parameter)")

    #******************************#
    #   compute parameter ranges   #
    #******************************#
    def compute_parameter_ranges(self, results):
        """
        Compute min/max/mean/std ranges for all fitted parameters.

        This is the key output for defining simulation parameter ranges:
        the ranges found during fitting of real in-vivo data can then be
        used as the parameter space for data simulation/augmentation.

        Args:
            results: Results dict from fit_nifti_list()

        Returns:
            dict with parameter ranges structured for use in Augmentrum/simulation
        """
        ranges = {
            'metadata': {
                'n_spectra': int(results['concentrations'].shape[0]),
                'method': results['method'],
                'ppmlim': list(results['ppmlim']),
                'date': datetime.now().isoformat(),
                'basis_names': results['basis_names'],
            },
            'concentrations': {},
            'signal_params': {},
        }

        concs = results['concentrations']
        basis_names = results['basis_names']

        # Concentration ranges per metabolite
        for i, name in enumerate(basis_names):
            col = concs[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                ranges['concentrations'][name] = {
                    'name': name,
                    'low_limit': float(np.min(valid)),
                    'up_limit': float(np.max(valid)),
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid)),
                    'median': float(np.median(valid)),
                    'p5': float(np.percentile(valid, 5)),
                    'p95': float(np.percentile(valid, 95)),
                    'n_valid': int(len(valid)),
                }

        # Signal parameter ranges (if available)
        if 'params' in results and 'param_names' in results:
            params = results['params']
            param_names = results['param_names']

            for i, pname in enumerate(param_names):
                if i >= params.shape[1]:
                    break
                col = params[:, i]
                valid = col[~np.isnan(col)]
                if len(valid) > 0:
                    ranges['signal_params'][pname] = {
                        'low_limit': float(np.min(valid)),
                        'up_limit': float(np.max(valid)),
                        'mean': float(np.mean(valid)),
                        'std': float(np.std(valid)),
                        'median': float(np.median(valid)),
                        'p5': float(np.percentile(valid, 5)),
                        'p95': float(np.percentile(valid, 95)),
                        'n_valid': int(len(valid)),
                    }

        # Extract Lorentzian-specific parameters (gamma per basis)
        # In FSL-MRS Voigt model: Lorentzian = gamma, Gaussian = sigma
        # For a pure Lorentzian lineshape, sigma ≈ 0
        gamma_params = {k: v for k, v in ranges['signal_params'].items()
                        if k.startswith('gamma')}
        sigma_params = {k: v for k, v in ranges['signal_params'].items()
                        if k.startswith('sigma')}
        if gamma_params:
            model_type = results.get('model', 'voigt')
            n_groups = results.get('n_groups', 1)
            n_basis = results.get('n_basis', 1)
            if model_type == 'free_shift':
                if n_groups == n_basis:
                    note = ('free_shift model with individual groups: '
                            'per-metabolite Lorentzian (gamma), per-metabolite Gaussian (sigma), '
                            'per-metabolite frequency shift (eps). '
                            'Sigma values are expected to be similar across metabolites '
                            '(Gaussian broadening arises from B0 shimming, a global effect). '
                            'Lorentzian FWHM (Hz) = 2 * gamma / pi.')
                else:
                    note = (f'free_shift model with {n_groups} group(s): '
                            f'gamma and sigma per group, eps per metabolite. '
                            'Lorentzian FWHM (Hz) = 2 * gamma / pi.')
            else:
                note = 'Lorentzian FWHM (Hz) = 2 * gamma / pi.'
            ranges['lineshape'] = {
                'type': model_type,
                'lorentzian_gamma': gamma_params,
                'gaussian_sigma': sigma_params,
                'note': note,
            }

        return ranges

    #************************#
    #   build JSON summary   #
    #************************#
    def _build_json_summary(self, results):
        """Build a human-readable JSON summary of fit results."""
        concs = results['concentrations']
        basis_names = results['basis_names']

        summary = {
            'augmentrum_version': 'FSL Fitting Framework v1.0',
            'date': datetime.now().isoformat(),
            'n_spectra': int(concs.shape[0]),
            'n_successful': int(np.sum(~np.isnan(concs[:, 0]))),
            'method': results['method'],
            'ppmlim': list(results['ppmlim']),
            'basis_names': basis_names,
            'concentration_summary': {},
        }

        for i, name in enumerate(basis_names):
            col = concs[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                summary['concentration_summary'][name] = {
                    'mean': round(float(np.mean(valid)), 4),
                    'std': round(float(np.std(valid)), 4),
                    'min': round(float(np.min(valid)), 4),
                    'max': round(float(np.max(valid)), 4),
                }

        if 'param_names' in results:
            summary['signal_param_names'] = results['param_names']

        return summary

    #************************#
    #   load saved results   #
    #************************#
    @staticmethod
    def load_results(save_path):
        """
        Load previously saved fitting results.

        Args:
            save_path: Path to results directory

        Returns:
            dict with:
                'results': Full results dict
                'ranges': Parameter ranges dict
        """
        loaded = {}

        # Load pickle results
        pkl_path = os.path.join(save_path, 'fit_results.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                loaded['results'] = pickle.load(f)
        else:
            # Reconstruct from numpy arrays
            loaded['results'] = {}
            concs_path = os.path.join(save_path, 'concentrations.npy')
            if os.path.exists(concs_path):
                loaded['results']['concentrations'] = np.load(concs_path)
            uncs_path = os.path.join(save_path, 'uncertainties.npy')
            if os.path.exists(uncs_path):
                loaded['results']['uncertainties'] = np.load(uncs_path)
            params_path = os.path.join(save_path, 'params.npy')
            if os.path.exists(params_path):
                loaded['results']['params'] = np.load(params_path)

        # Load parameter ranges
        ranges_path = os.path.join(save_path, 'parameter_ranges.json')
        if os.path.exists(ranges_path):
            with open(ranges_path, 'r') as f:
                loaded['ranges'] = json.load(f)

        # Load JSON summary
        json_path = os.path.join(save_path, 'fit_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                loaded['summary'] = json.load(f)

        print(f"✓ Loaded results from: {save_path}")
        if 'results' in loaded and 'concentrations' in loaded['results']:
            print(f"  Concentrations: {loaded['results']['concentrations'].shape}")
        if 'ranges' in loaded:
            print(f"  Parameter ranges: {len(loaded['ranges'].get('concentrations', {}))} metabolites")

        return loaded

    #*******************************#
    #   ranges to simulation defs   #
    #*******************************#
    @staticmethod
    def ranges_to_simulation_defs(ranges, margin=0.1):
        """
        Convert parameter ranges to simulation definition format
        (compatible with MRS project simulationDefs.py).

        Args:
            ranges: Parameter ranges dict (from compute_parameter_ranges)
            margin: Fractional margin to add to ranges (default: 10%)

        Returns:
            tuple: (conc_defs, signal_params) in MRS project format
        """
        conc_defs = {}
        for name, info in ranges.get('concentrations', {}).items():
            low = info['low_limit']
            high = info['up_limit']
            spread = high - low
            conc_defs[name] = {
                'name': name,
                'low_limit': max(0, low - margin * spread),
                'up_limit': high + margin * spread,
            }

        signal_params = {}
        for pname, info in ranges.get('signal_params', {}).items():
            if pname.startswith('conc_'):
                continue
            signal_params[pname] = {
                'low_limit': info['low_limit'],
                'up_limit': info['up_limit'],
                'mean': info['mean'],
                'std': info['std'],
            }

        return conc_defs, signal_params


#*************#
#   testing   #
#*************#
if __name__ == '__main__':
    print("FSL Fitting Framework - test")
    print("Use fit_cows_with_augmentrum.py for the full COWS fitting pipeline.")













