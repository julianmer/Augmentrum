####################################################################################################
#                                          brainbeats.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2024-08-25                                                                              #
#                                                                                                  #
# Purpose: Definition of data modules, taking care of loading and processing of the data.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import os
import scipy.io
import shutup; shutup.please()   # shut up warnings

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs, split
from fsl_mrs.utils.mrs_io import read_FID
from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

from tqdm import tqdm

# own
from augmentrum import Augmentrum
from augmentrum.processing.utils import (own_nifti_coil_combination,
                                         own_nifti_coil_combination_adaptive,
                                         resample_signal_fir, resample_signal_lp)


#****************************#
#   brainbeats data loader   #
#****************************#
def BrainBeatsData(data_dir, batch_size=16, seed=0, val_frac=0.1, test_frac=0.1,
                   n_coils=(1, None), n_averages=(1, None), pipelines=None,
                   modes=None, backend='pytorch', volatile=False, **kwargs):
    """
    Load BrainBeats data and create an Augmentrum instance.

    Args:
        data_dir: Path to BrainBeats data directory
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility
        val_frac: Validation fraction (default 0.1)
        test_frac: Test fraction (default 0.1)
        n_coils: Coil sampling range (min, max) or None
        n_averages: Average sampling range (min, max) or None
        pipelines: Custom pipelines dict or None for defaults
        modes: Sampling modes dict or None for defaults
        backend: Backend to use ('numpy', 'pytorch', etc.)
        volatile: If True, skip provenance logging
        **kwargs: Additional parameters for modules

    Returns:
        Augmentrum instance with BrainBeats data loaded
    """
    if modes is None:
        modes = {'train': 'random', 'val': 'deterministic', 'test': 'deterministic'}

    # Load all data once
    loader = BrainBeatsDataModule()
    data_dict = loader.load_all_niftis(data_dir)

    data, water = [], []
    for subject_id, sub_data in data_dict.items():
        for scan_id, scans in sub_data.items():
            data.append(scans['metab'])
            water.append(scans['wref'])

    # Create and return Augmentrum instance
    return Augmentrum(
        data=data,
        water=water,
        split_fractions={'val': val_frac, 'test': test_frac},
        pipelines=pipelines,
        modes=modes,
        backend=backend,
        batch_size=batch_size,
        seed=seed,
        volatile=volatile,
        n_coils=n_coils,
        n_averages=n_averages,
        **kwargs
    )


#**************************************************************************************************#
#                                    Class BrainBeatsDataModule                                    #
#**************************************************************************************************#
class BrainBeatsDataModule():

    #*****************************#
    #   convert labraw to nifti   #
    #*****************************#
    def labraw2nifti(self, data_dir, bw, cf, TE, TR, npoints, save_path=None):
        from loading.loadLABRAW import load_lab_raw, FIRdownsample

        for item in tqdm(os.listdir(data_dir)):
            # check if item is a directory and starts with 'B'
            if os.path.isdir(data_dir + '/' + item) and item.startswith('B'):
                # loop through folders
                for folder in os.listdir(data_dir + '/' + item):

                    if folder.startswith('2019') or folder.startswith('2020'):
                        # loop through files
                        for subfolder in os.listdir(data_dir + '/' + item + '/' + folder):
                            for file in os.listdir(data_dir + '/' + item + '/' + folder + '/' + subfolder):
                                try:
                                    if file.endswith('.raw.gz'):  # extract
                                        os.system(f'gunzip {data_dir}/{item}/{folder}/{subfolder}/{file}')
                                        file = file[:-3]

                                    if file.endswith('.raw'):
                                        data, info = load_lab_raw(f'{data_dir}/{item}/{folder}/{subfolder}/{file}')
                                        # data = FIRdownsample(data, npoints)

                                        # alternative resampling(s)
                                        data = resample_signal_lp(data, npoints, bw, axis=1)
                                        # data = resample_signal_fir(data, npoints)

                                        # get data and water
                                        data_met = np.transpose(data[:, :, 0, :], (1, 0, 2))
                                        data_wat = np.transpose(data[:, :, 1, :2], (1, 0, 2))

                                        # convert to nifti
                                        data_met = gen_nifti_mrs(data=data_met.reshape((1, 1, 1,) + data_met.shape),
                                                                 dwelltime=1 / bw,
                                                                 spec_freq=cf,
                                                                 nucleus='1H',
                                                                 dim_tags=['DIM_COIL', 'DIM_DYN', None],
                                                                 no_conj=True,
                                                                 )

                                        data_wat = gen_nifti_mrs(data=data_wat.reshape((1, 1, 1,) + data_wat.shape),
                                                                 dwelltime=1 / bw,
                                                                 spec_freq=cf,
                                                                 nucleus='1H',
                                                                 dim_tags=['DIM_COIL', 'DIM_DYN', None],
                                                                 no_conj=True,
                                                                 )

                                        # add header fields
                                        data_met.add_hdr_field('RepetitionTime', TR)
                                        data_met.add_hdr_field('EchoTime', TE)

                                        data_wat.add_hdr_field('RepetitionTime', TR)
                                        data_wat.add_hdr_field('EchoTime', TE)

                                        # save
                                        if save_path is not None:
                                            if 'pres' in file:
                                                session = 'PreS'
                                            elif 'posts2' in file:
                                                session = 'PostS2'
                                            elif 'posts' in file:
                                                session = 'PostS'
                                            else:
                                                continue

                                            if not os.path.exists(f'{save_path}/{item}/'):
                                                os.makedirs(f'{save_path}/{item}/')

                                            data_met.save(f'{save_path}/{item}/metab_{session}')
                                            data_wat.save(f'{save_path}/{item}/wref_{session}')

                                except:
                                    print(f"Issues with f'{data_dir}/{item}/{folder}/{subfolder}/{file}, skipping...")


    #********************************#
    #   convert spar sdat to nifti   #
    #********************************#
    def sdat2nifti(self, data_dir, bw, cf, TE, TR, save_path=None):
        from spec2nii.Philips.philips import read_sdat, read_spar

        for item in tqdm(os.listdir(data_dir)):
            # check if item is a directory and starts with 'B'
            if os.path.isdir(f'{data_dir}/{item}/') and item.startswith('B'):
                # loop through files
                for file in os.listdir(f'{data_dir}/{item}/'):
                    if file.endswith('.SPAR'):
                        try:
                            params = read_spar(f'{data_dir}/{item}/{file}')
                            data = read_sdat(f'{data_dir}/{item}/{file[:-4]}SDAT',
                                             params['samples'], params['rows'])
                        except:
                            print(f"Issues with f'{data_dir}/{item}/{file}, skipping...")
                            continue  # skip to the next session if files are not found

                        # convert to nifti
                        data = gen_nifti_mrs(data=data.reshape((1, 1, 1,) + data.shape),
                                             dwelltime=1 / bw,
                                             spec_freq=cf,
                                             nucleus='1H',
                                             dim_tags=[None, None, None],
                                             no_conj=True,
                                             )
                        # add header fields
                        data.add_hdr_field('RepetitionTime', TR)
                        data.add_hdr_field('EchoTime', TE)

                        # save
                        if save_path is not None:

                            if 'act' in file:
                                ty = 'metab'
                            elif 'ref' in file:
                                ty = 'wref'
                            else:
                                continue

                            if 'PreS' in file:
                                session = 'PreS'
                            elif 'PostS2' in file:
                                session = 'PostS2'
                            elif 'PostS' in file:
                                session = 'PostS'
                            else:
                                continue

                            if not os.path.exists(f'{save_path}/{item}/'):
                                os.makedirs(f'{save_path}/{item}/')

                            if ty == 'wref':
                                data.save(f'{save_path}/{item}/{ty}_{session}')


    #**************************#
    #   convert mat to nifti   #
    #**************************#
    def mat2nifti(self, data_dir, bw, cf, TE, TR, save_path=None):
        for item in tqdm(os.listdir(data_dir)):
            # check if item is a directory and starts with 'B'
            if os.path.isdir(data_dir + '/' + item) and item.startswith('B'):
                # loop through sessions (PreS, PostS, PostS2)
                for session in ["PreS", "PostS", "PostS2"]:
                    try:
                        mat_met = scipy.io.loadmat(f'{data_dir}/{item}/Brainbeats_{item}_{session}_metab.mat')
                        mat_wat = scipy.io.loadmat(f'{data_dir}/{item}/Brainbeats_{item}_{session}_wref.mat')
                        mat_met = mat_met['mat_met'].squeeze(-2)
                        mat_wat = mat_wat['mat_wat'].squeeze(-2)
                    except FileNotFoundError:
                        print(f"Files not found for {item} session {session}...")
                        continue  # skip to the next session if files are not found

                    # convert to nifti
                    data_met = gen_nifti_mrs(data=mat_met.reshape((1, 1, 1,) + mat_met.shape),
                                             dwelltime=1 / bw,
                                             spec_freq=cf,
                                             nucleus='1H',
                                             dim_tags=['DIM_COIL', 'DIM_DYN', None],
                                             no_conj=True,
                                             )

                    data_wat = gen_nifti_mrs(data=mat_wat.reshape((1, 1, 1,) + mat_wat.shape),
                                             dwelltime=1 / bw,
                                             spec_freq=cf,
                                             nucleus='1H',
                                             dim_tags=['DIM_COIL', 'DIM_DYN', None],
                                             no_conj=True,
                                             )

                    # add header fields
                    data_met.add_hdr_field('RepetitionTime', TR)
                    data_met.add_hdr_field('EchoTime', TE)

                    data_wat.add_hdr_field('RepetitionTime', TR)
                    data_wat.add_hdr_field('EchoTime', TE)

                    # save
                    if save_path is not None:
                        if not os.path.exists(f'{save_path}/{item}/'):
                            os.makedirs(f'{save_path}/{item}/')
                        data_met.save(f'{save_path}/{item}/metab_{session}')
                        data_wat.save(f'{save_path}/{item}/wref_{session}')



    #****************************************#
    #   segment processed data into blocks   #
    #****************************************#
    def write_blocks_preproc(self, data_dir, save_path=None):
        for item in tqdm(os.listdir(data_dir)):
            # check if item is a directory and starts with 'B'
            if os.path.isdir(data_dir + '/' + item) and item.startswith('B'):
                # loop through sessions (PreS, PostS, PostS2)
                for session in ["PreS", "PostS", "PostS2"]:
                    try:
                        data_met = read_FID(f'{data_dir}/{item}/preproc_{session}/metab.nii.gz')
                        data_wat = read_FID(f'{data_dir}/{item}/preproc_{session}/wref.nii.gz')
                        avg_wat = read_FID(f'{data_dir}/{item}/preproc_{session}/wref_av.nii.gz')

                    except FileNotFoundError:
                        print(f"Files not found for {item} session {session}...")
                        continue  # skip to the next session if files are not found

                    # process the files
                    if session == 'PreS':
                        data_met1, data_wat1 = self.process_nifti(data_met, data_wat)

                    elif session == 'PostS':
                        data_met2, data_met3 = split(data_met, 'DIM_DYN', 63)
                        _, data_met3 = split(data_met3, 'DIM_DYN', 15)
                        data_met3, data_met4 = split(data_met3, 'DIM_DYN', 63)
                        _, data_met4 = split(data_met4, 'DIM_DYN', 15)
                        data_met4, _ = split(data_met4, 'DIM_DYN', 63)

                        data_met2, data_wat2 = self.process_nifti(data_met2, data_wat)
                        data_met3, data_wat3 = self.process_nifti(data_met3, data_wat)
                        data_met4, data_wat4 = self.process_nifti(data_met4, data_wat)

                    elif session == 'PostS2':
                        data_met5, data_met6 = split(data_met, 'DIM_DYN', 63)

                        data_met5, data_wat5 = self.process_nifti(data_met5, data_wat)
                        data_met6, data_wat6 = self.process_nifti(data_met6, data_wat)

                # save the data
                if save_path is not None:
                    all_met = [data_met1, data_met2, data_met3, data_met4, data_met5, data_met6]
                    all_wat = [data_wat1, data_wat2, data_wat3, data_wat4, data_wat5, data_wat6]
                    for i in range(len(all_met)):
                        if not os.path.exists(f'{save_path}/{item}/block{i + 1}/'):
                            os.makedirs(f'{save_path}/{item}/block{i + 1}/')
                        all_met[i].save(f'{save_path}/{item}/block{i + 1}/metab')
                        all_wat[i].save(f'{save_path}/{item}/block{i + 1}/wref')


    #**********************************#
    #   segment the data into blocks   #
    #**********************************#
    def write_blocks(self, data_dir, save_path=None, process='custom', coil_combination='adaptive', limit_nsa=None):
        for item in tqdm(os.listdir(data_dir)):
            # check if item is a directory and starts with 'B'
            if os.path.isdir(data_dir + '/' + item) and item.startswith('B'):
                # loop through sessions (PreS, PostS, PostS2)
                for session in ["PreS", "PostS", "PostS2"]:
                    try:
                        data_met = read_FID(f'{data_dir}/{item}/metab_{session}.nii.gz')
                        data_wat = read_FID(f'{data_dir}/{item}/wref_{session}.nii.gz')

                    except FileNotFoundError:
                        print(f"Files not found for {item} session {session}...")
                        continue  # skip to the next session if files are not found

                    # process the files
                    if session == 'PreS':
                        data_met1, data_wat1 = data_met, data_wat

                    elif session == 'PostS':
                        data_met2, data_met3 = split(data_met, 'DIM_DYN', 63)
                        _, data_met3 = split(data_met3, 'DIM_DYN', 15)
                        data_met3, data_met4 = split(data_met3, 'DIM_DYN', 63)
                        _, data_met4 = split(data_met4, 'DIM_DYN', 15)
                        data_met4, _ = split(data_met4, 'DIM_DYN', 63)
                        data_wat2, data_wat3, data_wat4 = data_wat, data_wat, data_wat

                    elif session == 'PostS2':
                        data_met5, data_met6 = split(data_met, 'DIM_DYN', 63)
                        data_wat5, data_wat6 = data_wat, data_wat

                all_met = [data_met1, data_met2, data_met3, data_met4, data_met5, data_met6]
                all_wat = [data_wat1, data_wat2, data_wat3, data_wat4, data_wat5, data_wat6]

                # limit the number of averages if specified
                if limit_nsa is not None and isinstance(limit_nsa, int):
                    for i in range(len(all_met)):
                        if all_met[i].shape[5] > limit_nsa + 1:
                            all_met[i], _ = split(all_met[i], 'DIM_DYN', limit_nsa)
                        elif all_met[i].shape[5] == limit_nsa + 1:
                            continue  # do not split if the number of averages is equal to the limit
                        else:
                            print(f"Warning: {item} block {i + 1} has only {all_met[i].shape[5]} averages, "
                                  f"not splitting.")

                # save the data
                if save_path is not None:
                    sub_folder = f'preproc_{limit_nsa + 1}' if limit_nsa is not None and isinstance(limit_nsa, int) else 'preproc'
                    for i in range(len(all_met)):
                        if process.lower() == 'custom':
                            if not os.path.exists(f'{save_path}/{item}/block{i + 1}/{sub_folder}/'):
                                os.makedirs(f'{save_path}/{item}/block{i + 1}/{sub_folder}/')
                            all_met[i], all_wat[i] = self.process_nifti(all_met[i], all_wat[i],
                                                                        report=f'{save_path}/{item}/block{i + 1}/{sub_folder}/',
                                                                        coil_combination=coil_combination)
                            all_met[i].save(f'{save_path}/{item}/block{i + 1}/{sub_folder}/metab')
                            all_wat[i].save(f'{save_path}/{item}/block{i + 1}/{sub_folder}/wref')

                        elif process.lower() == 'fsl-mrs':
                            if not os.path.exists(f'{save_path}/{item}/block{i + 1}/'):
                                os.makedirs(f'{save_path}/{item}/block{i + 1}/')
                            all_met[i].save(f'{save_path}/{item}/block{i + 1}/metab')
                            all_wat[i].save(f'{save_path}/{item}/block{i + 1}/wref')

                            os.system(f'fsl_mrs_preproc'
                                      f' --data {save_path}/{item}/block{i + 1}/metab.nii.gz'
                                      f' --reference {save_path}/{item}/block{i + 1}/wref.nii.gz'
                                      f' --output {save_path}/{item}/block{i + 1}/{sub_folder}'
                                      f' --hlsvd --conjugate --overwrite --report')


    #**********************************#
    #   process the nifti data files   #
    #**********************************#
    def process_nifti(self, data_met, data_wat, report=None, coil_combination='adaptive'):

        data_met = proc.conjugate(data_met)
        data_wat = proc.conjugate(data_wat)

        avg_ref = proc.average(data_wat, 'DIM_DYN')

        if coil_combination.lower() == 'adaptive':
            data_met, data_wat = own_nifti_coil_combination_adaptive(data_met, data_wat, report=report)
        elif coil_combination.lower() == 'simple':
            data_met = own_nifti_coil_combination(data_met, avg_ref)
            data_wat = own_nifti_coil_combination(data_wat, avg_ref)
        elif coil_combination.lower() == 'fsl-mrs':
            noise = None
            no_prewhiten = False
            from fsl_mrs.utils.preproc.combine import estimate_noise_cov, CovarianceEstimationError
            stacked_data = []
            for dd, _ in data_met.iterate_over_dims(dim='DIM_COIL', iterate_over_space=True,
                                                    reduce_dim_index=True):
                stacked_data.append(dd)
            try:
                covariance = estimate_noise_cov(np.asarray(stacked_data))
            except CovarianceEstimationError:
                # if the attempt to form a covariance fails, disable prewhitening
                no_prewhiten = True

            data_met = proc.coilcombine(data_met, reference=avg_ref, report=report, noise=noise,
                                                  covariance=covariance, no_prewhiten=no_prewhiten)
            data_wat = proc.coilcombine(data_wat, reference=avg_ref, noise=noise,
                                                  covariance=covariance, no_prewhiten=no_prewhiten)

        data_met = proc.align(data_met, 'DIM_DYN', ppmlim=(0.2, 4.2), report=report)  # align phases
        data_wat = proc.align(data_wat, 'DIM_DYN', ppmlim=(0, 8))

        data_met, _ = proc.remove_unlike(data_met, report=report)  # remove outlier avergaes
        data_met = proc.average(data_met, 'DIM_DYN', report=report)  # combine averages
        data_wat = proc.average(data_wat, 'DIM_DYN')

        data_met = proc.ecc(data_met, data_wat, report=report)  # eddy current correction
        data_wat = proc.ecc(data_wat, data_wat)

        # data_met = proc.truncate_or_pad(data_met, -1, 'first', report=report)   # truncation
        # data_wat = proc.truncate_or_pad(data_wat, -1, 'first')

        data_met = proc.remove_peaks(data_met, [-0.15, -0.15], limit_units='ppm',
                                     report=report)  # remove residual water
        data_met = proc.shift_to_reference(data_met, 3.027, (2.9, 3.1), report=report)  # shift to ref

        data_met = proc.phase_correct(data_met, (2.9, 3.1), report=report)  # phase corretion
        data_wat = proc.phase_correct(data_wat, (4.55, 4.7), hlsvd=False)

        if report is not None:
            # remove report if it exists already
            if os.path.exists(report + 'mergedReports.html'): os.remove(report + 'mergedReports.html')

            import subprocess
            import glob
            htmlfiles = glob.glob(os.path.join(report, '*.html'))
            subprocess.call(['merge_mrs_reports', '-d', os.path.join(report, 'metab'),
                             '-o', report, '--delete'] + htmlfiles)

        return data_met, data_wat


    #*********************#
    #   load all niftis   #
    #*********************#
    def load_all_niftis(self, data_dir):
        from fsl_mrs.utils.mrs_io import read_FID

        sessions = ["PreS", "PostS", "PostS2"]
        data_dict = {}

        for item in tqdm(os.listdir(data_dir)):
            if os.path.isdir(os.path.join(data_dir, item)) and item.startswith('B'):
                data_dict[item] = {}
                for session in sessions:
                    metab_path = os.path.join(data_dir, item, f'metab_{session}.nii.gz')
                    wref_path = os.path.join(data_dir, item, f'wref_{session}.nii.gz')

                    if os.path.exists(metab_path) and os.path.exists(wref_path):
                        try:
                            data_met = read_FID(metab_path)
                            data_wat = read_FID(wref_path)
                            data_dict[item][session] = {'metab': data_met, 'wref': data_wat}
                        except Exception as e:
                            print(f"Error loading {item} {session}: {e}")
                    else:
                        print(f"Missing data for {item} session {session}, skipping.")
        return data_dict


    #*********************#
    #   load all blocks   #
    #*********************#
    def load_all_blocks(self, data_dir, internal='preproc', verbose=False):
        from fsl_mrs.utils.mrs_io import read_FID

        block_data = {}

        for item in tqdm(os.listdir(data_dir)):
            subj_path = os.path.join(data_dir, item)
            if os.path.isdir(subj_path) and item.startswith('B'):
                block_data[item] = {}
                for block_name in sorted(os.listdir(subj_path)):
                    block_path = os.path.join(subj_path, block_name, internal)
                    metab_file = os.path.join(block_path, 'metab.nii.gz')
                    wref_file = os.path.join(block_path, 'wref.nii.gz')

                    if os.path.exists(metab_file) and os.path.exists(wref_file):
                        try:
                            data_met = read_FID(metab_file)
                            data_wat = read_FID(wref_file)
                            block_data[item][block_name] = {'metab': data_met, 'wref': data_wat}
                        except Exception as e:
                            print(f"Error loading {item} {block_name}: {e}")
                    else:
                        if verbose: print(f"Missing files in {item} {block_name}, skipping.")

        return block_data

