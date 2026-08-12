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

from datetime import datetime

from fsl_mrs.core.nifti_mrs import NIFTI_MRS
from fsl_mrs.utils.preproc.nifti_mrs_proc import DimensionsDoNotMatch

from scipy import integrate

from nifti_mrs_plus import ops

# from suspect.processing.denoising import sliding_gaussian

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
#   own nifti eddy current correction   #
#***************************************#
def own_nifti_ecc(data, reference, report=None):
    """
    Eddy current correction for MRS data in the NIfTI format. Using the code from suspect.

    @param data -- The MRS data to be corrected.
    @param reference -- The reference data for the correction.
    @param report -- The report file (default=None).

    @returns -- The corrected MRS data.
    """
    from suspect.processing.denoising import sliding_gaussian

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

        ec_smooth = sliding_gaussian(np.unwrap(np.angle(ref)), 32)
        ecc = np.exp(-1j * ec_smooth)
        corrected_obj[idx] = dd * ecc

    if report is not None:
        raise NotImplementedError("Report generation not implemented yet for own_nifti_ecc")

    # update processing prov
    processing_info = f'{__name__}.own_nifti_ecc, '
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
def own_nifti_coil_combination_adaptive(data, reference=None, report=None):
    """
    Nifit wrapper for the adaptive coil combination.

    @param data -- The MRS data to be combined.
    @param reference -- The reference data for the combination (default=None).
    @param report -- The report file (default=None).

    @returns -- The combined MRS data.
    """
    if (reference is not None and data.shape[data.dim_position('DIM_COIL')] !=
            reference.shape[data.dim_position('DIM_COIL')]):
        raise DimensionsDoNotMatch('Reference and data coil dimension does not match.')

    combined_data = data.copy(remove_dim='DIM_COIL')
    combined_wat = reference.copy(remove_dim='DIM_COIL') if reference is not None else None

    for main, idx in data.iterate_over_spatial():
        main = np.reshape(main, data.shape[3:])   # prevent loosing dim when avg is 1

        # coil combination
        data_metab, data_wref = coil_combination_adaptive(main, reference[idx] if reference is not None else None)
        data_metab = np.reshape(data_metab, combined_data[idx].shape)   # adjust to lost dim when avg is 1

        # update data
        combined_data[idx] = data_metab
        if combined_wat is not None:
            combined_wat[idx] = data_wref

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
    """FSL-MRS FIDToSpec along the last axis: ortho fft, first point halved."""
    fids = np.array(fids, dtype=np.complex128)
    fids[..., 0] *= 0.5
    return np.fft.fftshift(np.fft.fft(fids, axis=-1, norm='ortho'), axes=-1)


def ppm_shift_axis(n, sw_hz, sf_mhz, shift=4.65):
    """The shifted ppm axis FSL-MRS builds for an n-point spectrum.

    The default *shift* is FSL-MRS's proton referencing constant
    (PPM_SHIFT['1H']), so the tensor path assumes 1H data.
    """
    return np.linspace(-sw_hz / 2, sw_hz / 2, n) / sf_mhz + shift


def ppm_window(n, sw_hz, sf_mhz, lim, shift=4.65):
    """First and last index of the ppm window *lim* (FSL-MRS limit_to_range)."""
    axis = ppm_shift_axis(n, sw_hz, sf_mhz, shift)
    first = int(np.argmin(np.abs(axis - lim[0])))
    last = int(np.argmin(np.abs(axis - lim[1])))
    return (first, last) if first <= last else (last, first)


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
