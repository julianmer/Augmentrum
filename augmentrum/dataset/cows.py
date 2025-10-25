####################################################################################################
#                                             cows.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-10-14                                                                              #
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
import scipy.io

# own
from augmentrum.dataset.base_dataset import BaseMRSDatasetLoader


#**************************************************************************************************#
#                                          Class COWSData                                          #
#**************************************************************************************************#
#                                                                                                  #
#  The data module to load the COWS dataset connected to the Augmentrum package.                   #
#                                                                                                  #
#**************************************************************************************************#
class COWSData(BaseMRSDatasetLoader):
    def __init__(self, data_dir, batch_size=16, seed=0, val_frac=0.1, test_frac=0.1,
                     n_coils=(1, None), n_averages=(1, None), pipelines=None, cache_det=True,
                     sampling_mode=None, to_tensor=True, **kwargs):
        if sampling_mode is None:
            sampling_mode = {'train': 'random', 'val': 'deterministic', 'test': 'deterministic'}

        if 'location' in kwargs:
            location = kwargs.pop('location')
        else:
            location = None
        if 'water_sup' in kwargs:
            water_sup = kwargs.pop('water_sup')
        else:
            water_sup = None

        # load all data once
        loader = COWSDataModule(data_dir=data_dir, location=location, water_sup=water_sup)

        # data, water, mm, mm_water, names = loader.load_mats()
        # data = [self.gen_nifti(fids=d['fid'][0][:, 0], bw=d['sw_h'][0].item(),
        #                        cf=d['sf'][0].item(), tr=2000, te=20) for d in data]
        # water = [self.gen_nifti(fids=w['fid'][0][:, 0], bw=w['sw_h'][0].item(),
        #                        cf=w['sf'][0].item(), tr=2000, te=20) for w in water]

        data, water, mm, mm_water, names = loader.load_twix()

        super().__init__(data=data, water=water, batch_size=batch_size, seed=seed,
                         val_frac=val_frac, test_frac=test_frac, n_coils=n_coils,
                         n_averages=n_averages, pipelines=pipelines, cache_det=cache_det,
                         sampling_mode=sampling_mode, to_tensor=to_tensor, **kwargs)

    def gen_nifti(self, fids, bw, cf, tr=None, te=None):
        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

        fids = gen_nifti_mrs(data=fids.reshape((1, 1, 1,) + fids.shape),
                             dwelltime=1 / bw,
                             spec_freq=cf,
                             nucleus='1H',
                             dim_tags=[None, None, None],
                             no_conj=True,
                             )

        # add header fields
        if tr is not None:
            fids.add_hdr_field('RepetitionTime', tr)
        if te is not None:
            fids.add_hdr_field('EchoTime', te)
        return fids


#**************************************************************************************************#
#                                        Class COWSDataModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load the COWS study data.                                                     #
#                                                                                                  #
#**************************************************************************************************#
class COWSDataModule():
    def __init__(self, data_dir, location=None, water_sup=None):
        self.data_dir = data_dir

        if location is None:
            self.location = ['OCCIPITAL', 'PARIETAL', 'PFL']
        elif isinstance(location, str):
            self.location = [location]

        if water_sup is None:
            self.water_sup = ['VAPOR', 'COWS7', 'COWS12']
        elif isinstance(water_sup, str):
            self.water_sup = [water_sup]

    def load_twix(self):
        data, refs, MM_data, MM_refs, names = [], [], [], [], []

        subjects = sorted(os.listdir(self.data_dir))
        for sub in subjects:
            sub_path = os.path.join(self.data_dir, sub)
            if not os.path.isdir(sub_path) or 'sub-' not in sub:
                continue

            # go through files
            for file in os.listdir(sub_path + '/mrs/sourcedata/'):
                if file.endswith('.dat'):
                    for location in self.location:
                        for ws in self.water_sup:
                            if location.lower() in file.lower() and ws.lower() in file.lower():
                                try:
                                    water, dat = self.load_twix_data(os.path.join(sub_path + '/mrs/sourcedata/', file))
                                    if 'metab' in file.lower():
                                        data.append(dat)
                                        refs.append(water)
                                    elif 'mm' in file.lower():
                                        MM_data.append(dat)
                                        MM_refs.append(water)
                                    names.append(f"{sub}_{location}_{ws}")
                                except Exception as e:
                                    print(f"Error loading {file}: {e}")

        return data, refs, MM_data, MM_refs, names

    def load_twix_data(self, file_path):
        from fsl_mrs.core.nifti_mrs import split
        from mapvbvd import mapVBVD
        from spec2nii.Siemens.twixfunctions import process_twix, examineTwix
        from augmentrum.processing.utils import safe_squeeze

        # call mapvbvd to load the twix file.
        twixObj = mapVBVD(os.path.join(file_path), quiet=True)
        # examineTwix(twixObj, file, 0)

        if isinstance(twixObj, list):
            twixObj = twixObj[-1]

        overrides = {'dims': (None, None, None),
                     'tags': (None, None, None)}
        file = os.path.basename(file_path)
        base_name = os.path.splitext(file)[0]
        nifti, fileoutNames = process_twix(twixObj, base_name, file, 'image', overrides, quiet=True)

        # extract water and data
        water, data = split(nifti[0], 'DIM_USER_0', 0)
        water.set_dim_tag(6, None)
        data.set_dim_tag(6, None)
        water, _ = split(water, 'DIM_DYN', 0)
        water, data = safe_squeeze(water), safe_squeeze(data)
        return water, data

    def load_mats(self):
        data, refs, MM_data, MM_refs, names = [], [], [], [], []

        subjects = sorted(os.listdir(self.data_dir))
        for sub in subjects:
            sub_path = os.path.join(self.data_dir, sub)
            if not os.path.isdir(sub_path):
                continue

            locations = sorted(os.listdir(sub_path))
            for loc in locations:
                if loc in self.location:
                    loc_path = os.path.join(sub_path, loc)
                    for file in os.listdir(loc_path):
                        if file.endswith('.mat'):
                            for ws in self.water_sup:
                                if ws in file:
                                    names.append(f"{sub}_{loc}_{ws}")
                                    if 'Water' in file:
                                        if 'Metab' in file:
                                            ref = scipy.io.loadmat(os.path.join(loc_path, file))
                                            refs.append(ref['exptDat'][0])
                                        elif 'MM' in file:
                                            ref = scipy.io.loadmat(os.path.join(loc_path, file))
                                            MM_refs.append(ref['exptDat'][0])
                                    else:
                                        if 'Metab' in file:
                                            dat = scipy.io.loadmat(os.path.join(loc_path, file))
                                            data.append(dat['exptDat'][0])
                                        elif 'MM' in file:
                                            dat = scipy.io.loadmat(os.path.join(loc_path, file))
                                            MM_data.append(dat['exptDat'][0])

        return data, refs, MM_data, MM_refs, names



#*************#
#   testing   #
#*************#
if __name__ == '__main__':
    # cows = COWSData(data_dir='data/COWS/COWS_mat/', to_tensor=False)
    cows = COWSData(data_dir='data/openneuro_ds006812/', to_tensor=False,
                    location='PFL', water_sup='VAPOR', conj=False,
                    coil_method='fsl-mrs')

    # example
    train_loader = cows.train_dataloader()
    x, x_ref = next(train_loader)
    import matplotlib.pyplot as plt

    for elem in x:
        print(elem.hdr_ext)a
        elem.plot()
        plt.show()