####################################################################################################
#                                          fmrsinpain.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-08                                                                              #
#                                                                                                  #
# Purpose: Definition of data modules, taking care of loading and processing of the data.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

# third-party
import numpy as np
import torch

from spec2nii.Philips.philips import read_sdat, read_spar
from spec2nii.Philips.philips_data_list import _read_list

from torch.utils.data import DataLoader

# own
from augmentrum.dataset.base_dataset import BaseMRSDatasetLoader
from augmentrum.utils.philips import read_Philips_data


#**************************************************************************************************#
#                                       Class fMRSinPainData                                       #
#**************************************************************************************************#
#                                                                                                  #
#  The data module to load the fMRSinPain dataset connected to the Augmentrum package.             #
#                                                                                                  #
#**************************************************************************************************#
class fMRSinPainData(BaseMRSDatasetLoader):
    def __init__(self, data_dir, batch_size=16, seed=0, val_frac=0.1, test_frac=0.1,
                     n_coils=(1, None), n_averages=(1, None), pipelines=None, cache_det=True,
                     sampling_mode=None, to_tensor=True, **kwargs):

        if sampling_mode is None:
            sampling_mode = {'train': 'random', 'val': 'deterministic', 'test': 'deterministic'}

        # load all data once
        loader = InVivoNSAModule(data_dir=data_dir, fMRS=False, coil_comb=True, verbose=0)

        data = [self.gen_nifti(d, bw=2000, cf=127.794504, tr=4000, te=23.46100044)
                for d in loader.data]
        water = [self.gen_nifti(w, bw=2000, cf=127.794504, tr=4000, te=23.46100044)
                 for w in loader.refs]

        super().__init__(data=data, water=water, batch_size=batch_size, seed=seed,
                         val_frac=val_frac, test_frac=test_frac, n_coils=n_coils,
                         n_averages=n_averages, pipelines=pipelines, cache_det=cache_det,
                         sampling_mode=sampling_mode, to_tensor=to_tensor, **kwargs)

    def gen_nifti(self, fids, bw, cf, tr=None, te=None):
        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

        # if len(fids.shape) == 2:   # add coil dim if missing
        #     fids = fids[:, np.newaxis]

        fids = gen_nifti_mrs(data=fids.reshape((1, 1, 1,) + fids.shape),
                             dwelltime=1 / bw,
                             spec_freq=cf,
                             nucleus='1H',
                             dim_tags=['DIM_DYN', None, None],
                             no_conj=True,
                             )

        # add header fields
        if tr is not None:
            fids.add_hdr_field('RepetitionTime', tr)
        if te is not None:
            fids.add_hdr_field('EchoTime', te)
        return fids


#**************************************************************************************************#
#                                       Class InVivoNSAModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load in-vivo data with individual NSA.                                        #
#                                                                                                  #
#**************************************************************************************************#
class InVivoNSAModule:

        #*************************#
        #   initialize instance   #
        #*************************#
        def __init__(self, data_dir, nums_test=None, fMRS=False, coil_comb=False, verbose=0):
            super().__init__()
            self.nums_test = nums_test
            self.fMRS = fMRS
            self.coil_comb = coil_comb

            self.data_dir = data_dir

            self.data = []
            self.refs = []

            # go through files in folder
            for file in os.listdir(data_dir):
                # if file is a folder
                if os.path.isdir(data_dir + '/' + file):
                    # go through files in sub-folder
                    for sub_file in os.listdir(data_dir + '/' + file):
                        # load fid
                        if sub_file.lower().split('.')[-1] == 'list':
                            fids, h2o = self.load_DATALIST_data(data_dir + '/' + file + '/' + sub_file)
                            if self.fMRS and len(fids.shape) > 2 + int(not self.coil_comb):  # fMRS dim: (samples, [coils], NSA, f-blocks)
                                self.data.append(fids)
                                self.refs.append(h2o)
                            if not self.fMRS and len(fids.shape) <= 2 + int(not self.coil_comb):   # (samples, [coils], NSA)
                                self.data.append(fids)
                                self.refs.append(h2o)

                        elif sub_file.lower().split('.')[-1] == 'sdat' or sub_file.lower().split('.')[-1] == 'spar':
                            if 'ref' in sub_file.lower():
                                h2o = self.load_SDATSPAR_data(data_dir + '/' + file + '/' + sub_file)
                                self.refs.append(h2o)
                            else:
                                fids = self.load_SDATSPAR_data(data_dir + '/' + file + '/' + sub_file)
                                self.data.append(fids)
                else:
                    # load fid
                    if file.lower().split('.')[-1] == 'list':
                        fids, h2o = self.load_DATALIST_data(data_dir + '/' + file)
                        if self.fMRS and len(fids.shape) > 2 + int(not self.coil_comb):
                            self.data.append(fids)
                            self.refs.append(h2o)
                        if not self.fMRS and len(fids.shape) <= 2 + int(not self.coil_comb):
                            self.data.append(fids)
                            self.refs.append(h2o)

                    elif file.lower().split('.')[-1] == 'sdat' or file.lower().split('.')[-1] == 'spar':
                        if 'ref' in file.lower():
                            h2o = self.load_SDATSPAR_data(data_dir + '/' + file)
                            self.refs.append(h2o)
                        else:
                            fids = self.load_SDATSPAR_data(data_dir + '/' + file)
                            self.data.append(fids)

        def load_DATALIST_data(self, path2data):
            df, num_dict, coord_dict, os_dict = _read_list(path2data[:-4] + 'list')
            sorted_data_dict = read_Philips_data(path2data[:-4] + 'data', df)

            fids = sorted_data_dict['STD_0'].squeeze()
            h2o = sorted_data_dict['STD_1'].squeeze()

            # naively combine channels
            if self.coil_comb:
                fids = fids.sum(1)  # fMRS in pain data has 27 channels but only one is used, as
                h2o = h2o.sum(1)    # a single-channel Transmit-Receive (T/R) head coil was used
            return fids, h2o

        def load_SDATSPAR_data(self, path2data):
            params = read_spar(path2data[:-4] + 'SPAR')
            data = read_sdat(path2data[:-4] + 'SDAT',
                             params['samples'],
                             params['rows'])
            return data

        def test_dataloader(self):
            data = torch.tensor(np.array(self.data))
            data = torch.fft.fft(data, dim=-2 - int(self.fMRS) - int(not self.coil_comb))
            data = torch.stack((data.real, data.imag), dim=1)
            return DataLoader(data[:self.nums_test], num_workers=4, batch_size=self.nums_test)

        def ref_dataloader(self):
            refs = torch.tensor(np.array(self.refs))
            refs = torch.fft.fft(refs, dim=-2 - int(self.fMRS) - int(not self.coil_comb))
            refs = torch.stack((refs.real, refs.imag), dim=1)
            return DataLoader(refs[:self.nums_test], num_workers=4, batch_size=self.nums_test)
