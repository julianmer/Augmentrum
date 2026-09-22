####################################################################################################
#                                              utils.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 21/06/22                                                                                #
#                                                                                                  #
# Purpose: Some helpful functions for processing MRS data are defined here.                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import nifti_mrs.utils as utils
import numpy as np
import warnings

from datetime import datetime


from scipy import integrate

from nifti_mrs_plus import ops

# own
from augmentrum import __version__


#***********************************************#
#   update nifti header with processing steps   #
#***********************************************#
def update_processing_prov(nmrs_obj, method, details):
    """
    Insert appropriate processing provenance information into the NIfTI-MRS header extension.

    https://github.com/wtclarke/fsl_mrs/blob/master/fsl_mrs/utils/preproc/nifti_mrs_proc.py

    @param nmrs_obj -- The NIFTI-MRS object which has been modified.
    @param method -- The processing method applied.
    @param details -- The details of the processing method.
    """
    # 1. Check for ProcessingApplied key and create if not present
    if 'ProcessingApplied' in nmrs_obj.hdr_ext:
        current_processing = nmrs_obj.hdr_ext['ProcessingApplied']
    else:
        current_processing = []

    # 2. Form object to append.
    prov_dict = {
        'Time': datetime.now().isoformat(sep='T', timespec='milliseconds'),
        'Program': 'Augmentrum',
        'Version': __version__,
        'Method': method,
        'Details': details}

    # 3. Append
    current_processing.append(prov_dict)
    nmrs_obj.add_hdr_field('ProcessingApplied', current_processing)


#*********************************#
#   phase correction of spectra   #
#*********************************#
def phaseCorrection(spectra):
    """
    Phase correction of the spectra.

    @param spectra -- The spectra.

    @returns -- The phase corrected spectra. Aligned to the maximum peak.
    """
    spectra = spectra[:, 0] + 1j * spectra[:, 1]
    maxIdx = np.argmax(np.abs(spectra), axis=1)
    phase = np.angle(np.take_along_axis(spectra, maxIdx[:, None], axis=1))
    spectra = spectra * np.exp(-1j * phase)
    return np.stack((np.real(spectra), np.imag(spectra)), axis=1)


#*******************************#
#   safe squeeze of NIFTI_MRS   #
#*******************************#
def safe_squeeze(mrs_obj, dims=None):
    """
    Return a new NIFTI_MRS with singleton higher-order dims removed.

    @param mrs_obj -- The NIFTI_MRS instance.
    @param dims -- None to remove all singleton higher dims,
                   or list/tuple of dim tags (e.g. 'DIM_COIL') or indices (4,5,6)
    @returns -- new NIFTI_MRS
    """
    from fsl_mrs.core.nifti_mrs import NIFTI_MRS
    # get the user facing data (note NIFTI_MRS.__getitem__ applies conjugation)
    data = mrs_obj[:]

    # ensure the array has the logical shape including implied singleton dims
    logical_shape = mrs_obj.shape
    if data.shape != logical_shape:
        data = data.reshape(logical_shape)

    # decide which dims to remove
    # only consider higher dims (4,5,6)
    all_higher = list(range(4, mrs_obj.ndim))
    if dims is None:
        dims_to_remove = [d for d in all_higher if logical_shape[d] == 1]
    else:
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        mapped = []
        for d in dims:
            if isinstance(d, str):
                mapped.append(mrs_obj._dim_tag_to_index(d))
            else:
                mapped.append(int(d))
        # only allow singleton dims
        dims_to_remove = [d for d in mapped if logical_shape[d] == 1]

    if not dims_to_remove:
        # nothing to do
        return NIFTI_MRS(data, header=mrs_obj.header)

    # remove axes from largest to smallest so indices remain valid
    dims_to_remove = sorted(set(dims_to_remove), reverse=True)

    for d in dims_to_remove:
        data = np.take(data, 0, axis=d)

    # update header extension
    new_hdr_ext = mrs_obj.hdr_ext.copy()
    # remove_dim_info expects dim index relative to 0 being dim_5 etc, so subtract 4
    for d in sorted(dims_to_remove):
        new_hdr_ext.remove_dim_info(d - 4)

    # produce a header patched with modified hdr ext
    new_header = mrs_obj.header.copy()
    new_header = utils.modify_hdr_ext(new_hdr_ext, new_header)

    new_obj = NIFTI_MRS(data, header=new_header)
    new_obj.hdr_ext = new_hdr_ext
    return new_obj


#***************************************#
#   sliding-gaussian smoothed phase     #
#***************************************#
def _sliding_gaussian_phase(ref: np.ndarray, width: int = 32) -> np.ndarray:
    """
    Smoothed unwrapped phase of a reference FID.

    Replicates "suspect.processing.denoising.sliding_gaussian" applied to
    "np.unwrap(np.angle(ref))" (edge-padded with 10-point edge means,
    correlated with a normalized Gaussian window), without needing the
    external "suspect" package installed. This is the same algorithm as
    "augmentrum.processing.raw_processing.RawProcessor._ecc_phase" (NumPy)
    and "augmentrum.processing.torch_engine.ecc_phase" (PyTorch) — those two
    are parity-tested against "suspect" itself, so this NIfTI-object path
    reuses the identical formula rather than a second, independently-derived
    one.

    Args:
        ref: Reference FID, "(..., T)" complex.
        width: Gaussian window width in samples.

    Returns:
        Smoothed unwrapped phase, same shape as "ref", real-valued.
    """
    phase = np.unwrap(np.angle(np.asarray(ref)), axis=-1)
    window = np.exp(-np.linspace(-3, 3, width) ** 2)
    window /= window.sum()
    offset = (width - 1) // 2
    left = np.broadcast_to(phase[..., :10].mean(axis=-1, keepdims=True),
                           phase.shape[:-1] + (offset,))
    right = np.broadcast_to(phase[..., -10:].mean(axis=-1, keepdims=True),
                            phase.shape[:-1] + (width - 1 - offset,))
    padded = np.concatenate([left, phase, right], axis=-1)
    return np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1) @ window


#***************************************#
#   nifti eddy current correction       #
#***************************************#
def nifti_ecc_smoothed(data, reference, report=None):
    """
    Eddy current correction for MRS data in the NIfTI format.

    @param data -- The MRS data to be corrected.
    @param reference -- The reference data for the correction.
    @param report -- The report file (default=None).

    @returns -- The corrected MRS data.
    """
    from fsl_mrs.utils.preproc.nifti_mrs_proc import DimensionsDoNotMatch

    if data.shape != reference.shape \
            and reference.ndim > 4:
        raise DimensionsDoNotMatch('Reference and data shape must match'
                                   ' or reference must be single FID.')

    corrected_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):

        if data.shape == reference.shape:
            # reference is the same shape as data, voxel-wise and spectrum-wise iteration
            ref = reference[idx]
        else:
            # only one reference FID, only iterate over spatial voxels
            ref = reference[idx[0], idx[1], idx[2], :]

        ec_smooth = _sliding_gaussian_phase(ref, 32)
        ecc = np.exp(-1j * ec_smooth)
        corrected_obj[idx] = dd * ecc

    if report is not None:
        raise NotImplementedError("Report generation not implemented yet for nifti_ecc_smoothed")

    # update processing prov
    processing_info = f'{__name__}.nifti_ecc_smoothed, '
    processing_info += f'reference={reference.filename}.'
    update_processing_prov(corrected_obj, 'Eddy current correction', processing_info)

    return corrected_obj


#********************************#
#   own nifti coil combination   #
#********************************#
def own_nifti_coil_combination(data, reference):
   """
    Simple coil combination for MRS data in the NIfTI format.
    Using amplitude and phase information from the water peak.

    @param data -- The MRS data to be combined.
    @param reference -- The reference data for the combination.

    @returns -- The combined MRS data.
   """
   combined_obj = data.copy(remove_dim='DIM_COIL')
   for ref, idx in reference.iterate_over_dims(dim='DIM_COIL',
                                               iterate_over_space=True,
                                               reduce_dim_index=False):

       # coil-based weighted average and phase correction
       water_amps = integrate.trapezoid(np.abs(ref), axis=0)
       water_amps = (water_amps / np.sum(water_amps))[np.newaxis, :, np.newaxis]
       water_phases = np.angle(integrate.trapezoid(ref, axis=0))[np.newaxis, :, np.newaxis]
       data_metab = data[idx] * water_amps / np.exp(1j * water_phases)
       combined_obj[idx] = np.sum(data_metab, axis=-2)   # sum over coils

   # update processing prov
   processing_info = f'{__name__}.coil_combination, '
   processing_info += f'reference={reference.filename}.'
   update_processing_prov(combined_obj, 'Coil combination', processing_info)

   return combined_obj


#*********************************#
#   estimate coil sensitivities   #
#*********************************#
def estimate_csm(data):
    """
    Estimate the coil sensitivity maps (CSM) from the reference data.
    Adapted from FID-A.

    @param data -- The reference data.

    @returns -- The coil sensitivity maps.
    """
    s_raw = data / (np.sqrt(np.sum(data * np.conj(data), axis=0)) + np.finfo(float).eps)
    Rs = np.einsum('ij,ik->jk', s_raw, np.conj(s_raw))
    csm, _ = eig_power(Rs)
    return csm


#*****************************#
#   eigenvalue power method   #
#*****************************#
def eig_power(R):
    """
    Eigenvalue power method for the coil sensitivity maps (CSM)
    from the autocorrelation matrix.

    @param R -- The reference data.

    @returns -- The coil sensitivity maps.
    """
    rows, cols = R.shape
    N_iterations = 2
    v = np.ones((rows, cols), dtype=complex)
    for _ in range(N_iterations):
        v = np.dot(R, v)
        d = np.sqrt(np.sum(np.abs(v) ** 2, axis=0))
        d[d <= np.finfo(float).eps] = np.finfo(float).eps
        v = v / d
    p1 = np.angle(np.conj(v[:, 0]))
    v = v * np.exp(1j * p1)[:, None]
    return np.conj(v), d


#*********************************#
#   coil combination (adaptive)   #
#*********************************#
def coil_combination_adaptive(data, water=None):
    """
    Coil combination using amplitude and phase information from the water peak.
    Adapted from CIBM.

    @param data -- The MRS data to be combined.
    @param water -- The reference data for the combination (default=None).

    @returns -- The combined MRS data.
    """
    ref = water if water is not None and water.size != 0 else data
    if ref is data:
        print("No water reference provided, using metabolite data as reference")

    # compute the coil sensitivity maps
    ref = np.mean(ref, axis=2)
    phase = np.exp(-1j * np.angle(ref))
    csm = estimate_csm(ref * phase)[:, 0]
    csmsq = np.sum(csm * np.conj(csm), axis=0)
    csm[csm < np.finfo(float).eps] = 1

    def combine(data):
        combined = (np.sum(np.conj(csm)[None, :, None] * data * phase[..., None], axis=1) /
                    (csmsq + np.finfo(float).eps))
        return combined

    if water is None:
        return combine(data), None
    else:
        return combine(data), combine(water)


#****************************************#
#   nifit wrapper for coil combination   #
#****************************************#
def nifti_coil_combination_adaptive(data, reference=None, report=None):
    """
    Nifit wrapper for the adaptive coil combination.

    @param data -- The MRS data to be combined.
    @param reference -- The reference data for the combination (default=None).
    @param report -- The report file (default=None).

    @returns -- The combined MRS data.
    """
    from fsl_mrs.utils.preproc.nifti_mrs_proc import DimensionsDoNotMatch
    if (reference is not None and data.shape[data.dim_position('DIM_COIL')] !=
            reference.shape[data.dim_position('DIM_COIL')]):
        raise DimensionsDoNotMatch('Reference and data coil dimension does not match.')

    combined_data = data.copy(remove_dim='DIM_COIL')
    combined_wat = reference.copy(remove_dim='DIM_COIL') if reference is not None else None

    for main, idx in data.iterate_over_spatial():
        main = np.reshape(main, data.shape[3:])   # prevent loosing dim when avg is 1

        # The water is its own acquisition: a single transient has no
        # dynamic axis at all, so it is given one to average over.
        wref = None
        if reference is not None:
            wref = np.reshape(reference[idx], reference.shape[3:])
            if wref.ndim == 2:
                wref = wref[..., None]

        # coil combination
        data_metab, data_wref = coil_combination_adaptive(main, wref)
        data_metab = np.reshape(data_metab, combined_data[idx].shape)   # adjust to lost dim when avg is 1

        # update data
        combined_data[idx] = data_metab
        if combined_wat is not None:
            combined_wat[idx] = np.reshape(data_wref, combined_wat[idx].shape)

    # plot
    if report is not None:
        for main, idx in data.iterate_over_dims(dim='DIM_COIL',
                                                iterate_over_space=True,
                                                reduce_dim_index=False):

            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            if all([ii == slice(None, None, None) or ii == 0 for ii in idx]):  # first index
                fig = combine_FIDs_report(main,
                                          combined_data[:].mean(-1).squeeze(),
                                          data.bandwidth,
                                          data.spectrometer_frequency[0],
                                          data.nucleus[0],
                                          ncha=data.shape[data.dim_position('DIM_COIL')],
                                          ppmlim=(0.0, 6.0),
                                          method='adaptive',
                                          dim='DIM_COIL',
                                          html=report)

    # update processing prov
    processing_info = f'{__name__}.coil_combination, '
    processing_info += f'reference={reference.filename if reference is not None else "None"}.'
    update_processing_prov(combined_data, 'Coil combination', processing_info)

    return combined_data, combined_wat


#*****************************#
#   resample and FIR filter   #
#*****************************#
def resample_signal_fir(data, npoints, ntabs=12):
    """
    Resample the data to npoints using a FIR filter.

    @param data -- The data to be resampled.
    @param npoints -- The number of points to resample to.
    @param ntabs -- The number of filter taps (default=12).

    @returns -- The resampled data.
    """
    from scipy.signal import firwin, lfilter

    # FIR filter design parameters
    downsample_factor = data.shape[1] // npoints
    cutoff_frequency = 1 / (2 * downsample_factor)   # Nyquist

    # FIR filter
    fir_coeffs = firwin(ntabs, cutoff_frequency, window='hamming')
    data = lfilter(fir_coeffs, 1.0, data.squeeze(), axis=1)[:, ::downsample_factor, ...]

    return data


#*********************#
#   resample signal   #
#*********************#
def resample_signal_lp(data, npoints, bandwidth, axis=1):
    """
    Resample the signal to a new sampling frequency.

    @param data -- The signal to be resampled.
    @param npoints -- The number of points to resample to.
    @param bandwidth -- The desired bandwidth.
    @param axis -- The axis along which to resample (default=1).

    @returns -- The resampled signal.
    """
    from scipy.signal import resample, butter, filtfilt

    # low-pass filter
    nyquist = data.shape[axis] / 2
    cutoff = bandwidth / nyquist  # normalized cutoff frequency
    b, a = butter(4, cutoff, btype='low')  # 4th-order Butterworth filter

    # apply the filter and resample along axis 1
    def process_signal(single_signal):
        filtered_signal = filtfilt(b, a, single_signal)
        resampled_signal = resample(filtered_signal, npoints)
        return resampled_signal

    return np.apply_along_axis(process_signal, axis=axis, arr=data.squeeze())


#******************************#
#   fsl-mrs axis conventions   #
#******************************#
def fid_to_spec(fids):
    """
    FSL-MRS FIDToSpec along the last axis: ortho fft, first point halved; in the precision of
    *fids* (NumPy's FFT computes in double, so the result is put back).
    """
    fids = np.array(fids)
    fids = fids.astype(np.result_type(fids.dtype, np.complex64), copy=False)
    fids[..., 0] *= 0.5
    spec = np.fft.fftshift(np.fft.fft(fids, axis=-1, norm='ortho'), axes=-1)
    return spec.astype(fids.dtype, copy=False)


#: fsl_mrs.utils.constants.PPM_SHIFT, copied so the axis helpers work without FSL-MRS.
_PPM_SHIFT = {'1H': 4.65, '2H': 4.8, '13C': 0.0, '31P': 0.0}


def ppm_reference(nucleus='1H'):
    """
    The ppm at which FSL-MRS / NIfTI-MRS place the carrier (0 Hz) for *nucleus*.

    Protons are referenced to water at 4.65 ppm; 13C and 31P to 0 ppm. Every
    module that places a feature at a ppm uses this, so a user's ppm values mean
    the same thing here as on an FSL-MRS plot.

    Args:
        nucleus: NIfTI-MRS nucleus string, e.g. '1H'. None means 1H.

    Returns:
        The reference shift in ppm. An unknown nucleus gives 0.0 with a warning,
        which is the right answer for most X-nuclei and loud for the rest.
    """
    if nucleus is None:
        nucleus = '1H'
    try:
        from fsl_mrs.utils.constants import PPM_SHIFT
    except ImportError:                                   # pragma: no cover
        PPM_SHIFT = _PPM_SHIFT

    key = str(nucleus).strip().upper()
    for name, shift in PPM_SHIFT.items():
        if name.upper() == key:
            return float(shift)

    import warnings
    warnings.warn(f"No ppm reference known for nucleus {nucleus!r}; using 0.0 ppm.")
    return 0.0


def ppm_axis(n, sw_hz, sf_mhz, nucleus='1H'):
    """
    The FSL-MRS ppm of every bin of the spectrum "fftshift(ifft(fid))".

    That is the spectrum the spectral modules work on (see "DomainTransform").
    FSL-MRS displays "fftshift(fft(fid))" instead and labels its bins with
    "linspace(-sw/2, sw/2, n) / sf + reference" ("MRS.getAxes"), a grid whose
    step is sw/(n-1) rather than sw/n - up to one bin off the fftfreq grid at the
    edges. Since "ifft" mirrors "fft", bin j here is bin (-j) mod n there, so this
    axis is FSL's read backwards: it descends in ppm. The Nyquist bin (j = 0) is
    given its +sw/2 alias instead of FSL's -sw/2 label, which keeps the axis
    monotonic; every other bin carries exactly the ppm FSL-MRS would print for
    it. A feature built at a ppm on this axis therefore lands at that ppm on an
    FSL-MRS plot, which is what users' ppm values refer to.

    Args:
        n: Number of spectral points.
        sw_hz: Spectral width (bandwidth) in Hz.
        sf_mhz: Spectrometer frequency in MHz.
        nucleus: NIfTI-MRS nucleus string; sets the reference via "ppm_reference".

    Returns:
        A "(n,)" float array, descending in ppm.
    """
    ref = ppm_reference(nucleus)
    if n < 2:
        return np.full(int(n), ref)
    j = np.arange(n, dtype=np.float64)
    return ref + (sw_hz / 2.0 - (j - 1.0) * sw_hz / (n - 1.0)) / float(sf_mhz)


def ppm_shift_axis(n, sw_hz, sf_mhz, shift=None):
    """
    The shifted ppm axis FSL-MRS builds for an n-point "fftshift(fft(fid))" spectrum.

    Identical to "MRS.getAxes()" ("linspace(-sw/2, sw/2, n) / sf + shift"), so it
    goes with "fid_to_spec". "ppm_axis" is the same labelling for the mirrored
    "fftshift(ifft(fid))" spectrum the modules use.

    Args:
        n: Number of spectral points.
        sw_hz: Spectral width in Hz.
        sf_mhz: Spectrometer frequency in MHz.
        shift: Reference ppm; None is the proton reference (4.65 ppm).
    """
    if shift is None:
        shift = ppm_reference('1H')
    return np.linspace(-sw_hz / 2, sw_hz / 2, n) / sf_mhz + shift


def ppm_window(n, sw_hz, sf_mhz, lim, shift=None):
    """First and last index of the ppm window *lim* on "ppm_shift_axis" (FSL-MRS limit_to_range)."""
    axis = ppm_shift_axis(n, sw_hz, sf_mhz, shift)
    first = int(np.argmin(np.abs(axis - lim[0])))
    last = int(np.argmin(np.abs(axis - lim[1])))
    return (first, last) if first <= last else (last, first)


#***********************#
#   causal lineshapes   #
#***********************#
def causal_lineshape(ppm, center_ppm, lorentz_ppm=0.0, gauss_ppm=0.0):
    """
    The spectrum of a causal resonance at *center_ppm* on the axis *ppm*, unit peak real height.

    A resonance is a decaying complex exponential in the FID,
    "exp(2 pi i f t) exp(-pi lb t) exp(-(pi gb t)^2 / (4 ln 2))", whose spectrum
    is the Lorentzian of FWHM lb, the Gaussian of FWHM gb, or their convolution,
    a Voigt. It is built there and transformed as "DomainTransform" does
    ("fftshift(ifft(fid))"), which is what keeps it causal: a lineshape drawn
    directly on the axis is real, so its FID is two-sided with half of it wrapped
    to the end of the acquisition, where it rings whenever the FID is zero-filled
    or truncated. Everything is expressed in bins read off the axis - the
    frequency is the one that peaks on the bin the axis labels *center_ppm*, and a
    width is its FWHM in ppm over the axis step - so a feature lands and measures
    exactly where its frequency-domain twin would, whichever way the axis runs.

    Args:
        ppm: The ppm of every bin of "fftshift(ifft(fid))", uniform apart from the
            Nyquist alias "ppm_axis" puts in bin 0.
        center_ppm: Where the peak sits.
        lorentz_ppm: Lorentzian FWHM in ppm, 0 for none.
        gauss_ppm: Gaussian FWHM in ppm, 0 for none. With neither width the
            resonance does not decay and its spectrum is the Dirichlet kernel.

    Returns:
        A "(n,)" complex array whose real part peaks at 1.
    """
    ppm = np.asarray(ppm, dtype=np.float64)
    n = ppm.size
    if n < 2:
        return np.ones(n, dtype=np.complex128)

    # The median step survives the alias in bin 0; its sign carries the direction.
    step = float(np.median(np.diff(ppm)))
    centre_bin = n // 2 + (float(center_ppm) - ppm[n // 2]) / step
    t = np.arange(n, dtype=np.float64)

    # Bin j of fftshift(ifft(fid)) holds (n//2 - j)/n cycles per sample, and a width
    # of w bins is exp(-pi w t / n) per sample for a Lorentzian.
    fid = np.exp(2j * np.pi * (n // 2 - centre_bin) / n * t)
    if lorentz_ppm > 0:
        fid = fid * np.exp(-np.pi * (lorentz_ppm / abs(step)) * t / n)
    if gauss_ppm > 0:
        fid = fid * np.exp(-(np.pi * (gauss_ppm / abs(step)) * t / n) ** 2 / (4.0 * np.log(2.0)))

    spectrum = np.fft.fftshift(np.fft.ifft(fid))
    return spectrum / np.max(np.real(spectrum))


def _causal_lineshapes_device(winding, lorentz, gauss, step, n, like):
    """"causal_lineshapes" past its per-row winding factors, in float64 on *like*'s device."""
    import torch
    t = device_axis(('index', n), lambda: np.arange(n, dtype=np.float64), like)
    fid = torch.exp(device_values(winding, like).reshape(-1, 1) * t)
    lorentz_d, gauss_d = device_values(lorentz, like), device_values(gauss, like)
    if np.any(lorentz > 0):
        damped = fid * torch.exp(-np.pi * (lorentz_d / abs(step)) * t / n)
        fid = torch.where(lorentz_d > 0, damped, fid)
    if np.any(gauss > 0):
        damped = fid * torch.exp(-(np.pi * (gauss_d / abs(step)) * t / n) ** 2
                                 / (4.0 * np.log(2.0)))
        fid = torch.where(gauss_d > 0, damped, fid)
    spectrum = torch.fft.fftshift(torch.fft.ifft(fid, dim=-1), dim=-1)
    return spectrum / torch.amax(spectrum.real, dim=-1, keepdim=True)


def causal_lineshapes(ppm, center_ppm, lorentz_ppm=0.0, gauss_ppm=0.0, like=None):
    """
    "causal_lineshape" for a batch of resonances at once, "(batch, n)".

    Row b is "causal_lineshape(ppm, center_ppm[b], lorentz_ppm[b], gauss_ppm[b])"
    bit for bit: the per-row factors are formed exactly as the scalar ones
    are, a width is applied only to the rows where it is positive, and the
    transforms run row by row on the batch. It exists so that a module drawing
    a resonance per sample pays for one set of array operations instead of one
    per sample.

    Args:
        ppm: The ppm of every bin, as for "causal_lineshape".
        center_ppm: "(batch,)" peak positions.
        lorentz_ppm: Lorentzian FWHMs in ppm, a scalar or "(batch,)"; 0 for none.
        gauss_ppm: Gaussian FWHMs in ppm, a scalar or "(batch,)"; 0 for none.
        like: A tensor; a CUDA one has the rows built on its device ("on_cuda").

    Returns:
        A "(batch, n)" complex array whose rows' real parts peak at 1 (complex128,
        on *like*'s device when built there).
    """
    ppm = np.asarray(ppm, dtype=np.float64)
    centers = np.asarray(center_ppm, dtype=np.float64).reshape(-1)
    batch, n = centers.size, ppm.size
    if n < 2:
        return np.ones((batch, n), dtype=np.complex128)
    lorentz = np.broadcast_to(np.asarray(lorentz_ppm, dtype=np.float64), (batch,))[:, None]
    gauss = np.broadcast_to(np.asarray(gauss_ppm, dtype=np.float64), (batch,))[:, None]

    step = float(np.median(np.diff(ppm)))
    t = np.arange(n, dtype=np.float64)

    # The winding factor goes through Python complex arithmetic as in the scalar
    # version (which divides where NumPy's complex division would multiply by
    # the reciprocal), so every row starts from the same complex number.
    winding = np.array([2j * np.pi * (n // 2 - (n // 2 + (float(c) - ppm[n // 2]) / step)) / n
                        for c in centers])
    if like is not None and on_cuda(like):
        return _causal_lineshapes_device(winding, lorentz, gauss, step, n, like)
    fid = np.exp(winding[:, None] * t)
    if np.any(lorentz > 0):
        damped = fid * np.exp(-np.pi * (lorentz / abs(step)) * t / n)
        fid = np.where(lorentz > 0, damped, fid)
    if np.any(gauss > 0):
        damped = fid * np.exp(-(np.pi * (gauss / abs(step)) * t / n) ** 2 / (4.0 * np.log(2.0)))
        fid = np.where(gauss > 0, damped, fid)

    spectrum = np.fft.fftshift(np.fft.ifft(fid, axis=-1), axes=-1)
    return spectrum / np.max(np.real(spectrum), axis=-1, keepdims=True)


#***********************#
#   per-sample values   #
#***********************#
def batch_profile(profile, ndim):
    """
    A "(batch, N)" profile shaped to broadcast over a rank-*ndim* "(batch, ..., N)" array.

    Profiles are built one per sample and must reach every coil and transient
    of that sample alike, so the sample axis stays in front and everything in
    between is a singleton.

    Args:
        profile: "(batch, N)" array.
        ndim: Rank of the array it will be added to or multiplied with.

    Returns:
        The profile reshaped to "(batch, 1, ..., 1, N)"; for a 1-D target, its
        single row.
    """
    profile = np.asarray(profile)
    if ndim <= 1:
        return profile[0]
    return profile.reshape((profile.shape[0],) + (1,) * (ndim - 2) + (profile.shape[-1],))


def per_sample_factor(value, ndim, like):
    """
    A scalar as a float, or a per-sample vector as "(batch, 1, ..., 1)" on *like*'s backend.

    The pipeline hands a module either one value for the batch or one per
    sample; this makes the two cases one multiply at the call site.

    Args:
        value: Scalar, or a "(batch,)" vector of per-sample values.
        ndim: Rank of the tensor the factor multiplies.
        like: Tensor whose backend and dtype the vector adopts.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return to_backend(arr.reshape((-1,) + (1,) * (ndim - 1)), like)


def on_cuda(like):
    """
    Whether *like* is a CUDA tensor, whose per-batch profiles are then built on its device.

    A module that builds a "(batch, N)" profile in NumPy pays for the host
    arithmetic and for an upload every batch. On a CUDA tensor the same float64
    arithmetic runs on the device from the host-drawn parameters instead, so
    only those parameters travel; the result agrees with the NumPy build to
    float64 rounding, far below the precision the profile is applied in.
    """
    return ops.is_torch(like) and like.device.type == 'cuda'


def device_values(values, like):
    """Host-drawn parameter *values* on *like*'s CUDA device, as float64 (complex128 if complex)."""
    from augmentrum.processing.torch_engine import upload
    arr = np.asarray(values)
    arr = arr.astype(np.complex128 if np.iscomplexobj(arr) else np.float64, copy=False)
    return upload(arr, like.device)


def device_axis(key, build, like):
    """A float64 axis built on the host once (NumPy, as the host path builds it) and kept on the device."""
    from augmentrum.processing.torch_engine import constant
    return constant(key, lambda: np.asarray(build(), dtype=np.float64), like.device)


def to_backend(param, like, dtype=None):
    """
    "ops.match_backend" - or, with *dtype*, "ops.asarray_like" - without the wait.

    Uploading a host array to a CUDA tensor makes the host wait for all the
    work queued on the device before it. Going through pinned memory without
    blocking lets a pipeline queue a whole batch and the device run it in one
    go; the values, dtype and device are the same.

    Args:
        param: A NumPy parameter array.
        like: The tensor whose backend and device it goes to.
        dtype: Target dtype name; None adopts *like*'s.
    """
    if ops.is_torch(like) and like.device.type == 'cuda':
        import torch
        from augmentrum.processing.torch_engine import upload
        target = getattr(torch, dtype) if dtype is not None else like.dtype
        if isinstance(param, torch.Tensor):      # built on the device already ("on_cuda")
            return param.to(like.device).to(target)
        return upload(param, like.device).to(target)
    if dtype is not None:
        return ops.asarray_like(like, param, dtype=dtype)
    return ops.match_backend(param, like)


#********************#
#   axis shuffling   #
#********************#
def move_axis(x, src, dst):
    """Move one axis of *x* from *src* to *dst*, like numpy.moveaxis."""
    rank = len(ops.shape(x))
    src, dst = src % rank, dst % rank
    order = [d for d in range(rank) if d != src]
    order.insert(dst, src)
    return ops.transpose(x, order)
