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
from augmentrum import Augmentrum


#**********************#
#   cows data loader   #
#**********************#
def COWSData(data_dir, batch_size=16, seed=0, val_frac=0.1, test_frac=0.1,
             n_coils=(1, None), n_averages=(1, None), pipelines=None,
             modes=None, backend='pytorch', volatile=False, **kwargs):
    """
    Load COWS data and create an Augmentrum instance.

    Args:
        data_dir: Path to COWS data directory
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
        **kwargs: Additional parameters (location, water_sup, etc.)

    Returns:
        Augmentrum instance with COWS data loaded
    """
    if modes is None:
        modes = {'train': 'random', 'val': 'deterministic', 'test': 'deterministic'}

    if 'location' in kwargs:
        location = kwargs.pop('location')
    else:
        location = None
    if 'water_sup' in kwargs:
        water_sup = kwargs.pop('water_sup')
    else:
        water_sup = None

    # Load all data once
    loader = COWSDataModule(data_dir=data_dir, location=location, water_sup=water_sup)
    data, water, mm, mm_water, names = loader.load_twix()

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
#                                       Class COWSDataModule                                       #
#**************************************************************************************************#
class COWSDataModule():
    def __init__(self, data_dir, location=None, water_sup=None):
        self.data_dir = data_dir

        if location is None:
            self.location = ['OCCIPITAL', 'PARIETAL', 'PFL', 'PFC']
        elif isinstance(location, str):
            # Accept PFL or PFC as aliases for the prefrontal region
            if location.upper() in ('PFL', 'PFC'):
                self.location = ['PFL', 'PFC']
            else:
                self.location = [location]
        else:
            self.location = location

        if water_sup is None:
            self.water_sup = ['VAPOR', 'COWS7', 'COWS12']
        elif isinstance(water_sup, str):
            self.water_sup = [water_sup]

    def load_twix(self):
        data, refs, MM_data, MM_refs, names = [], [], [], [], []
        self.load_failures = []  # (filename, error_msg)

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
                                    names.append(f"{sub}_{location.replace('PFC', 'PFL')}_{ws}")
                                except Exception as e:
                                    self.load_failures.append((file, str(e)))
                                    print(f"Error loading {file}: {e}")

        return data, refs, MM_data, MM_refs, names
    def load_twix_data(self, file_path, remove_oversampling=True):
        from fsl_mrs.core.nifti_mrs import split
        from spec2nii.Siemens.twixfunctions import process_twix, examineTwix
        from augmentrum.processing.utils import safe_squeeze

        try:
            from mapvbvd import mapVBVD
        except ImportError as error:
            raise ImportError(
                "Reading Siemens twix files needs mapvbvd. Install it with "
                "\"pip install pymapvbvd\", or point this at data already "
                "converted to NIfTI-MRS."
            ) from error

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

        # ── Remove Siemens 2x oversampling ────────────────────────────────────
        # Raw TWIX: 4096 pts / 8000 Hz (dt=125 us).
        # True acquisition: 2048 pts / 4000 Hz (dt=250 us).
        # Decimate by 2 along spectral axis (axis 0 of squeezed data).
        if remove_oversampling:
            water = self._remove_oversampling(water)
            data  = self._remove_oversampling(data)

        water, data = safe_squeeze(water), safe_squeeze(data)
        return water, data
    @staticmethod
    def _remove_oversampling(nifti_obj, factor=2):
        """Remove Siemens oversampling by decimating spectral points by `factor`.

        Takes every `factor`-th point along the spectral (4th NIfTI) dimension,
        and rebuilds the NIFTI_MRS object with the correct dwell time.

        Args:
            nifti_obj: NIFTI_MRS object
            factor: Oversampling factor to remove (default 2)

        Returns:
            New NIFTI_MRS object with oversampling removed.
        """
        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

        arr = np.asarray(nifti_obj[:])              # (1,1,1, npts, ...)
        old_dt = float(nifti_obj.dwelltime)
        hdr = nifti_obj.hdr_ext

        # Decimate along the spectral axis (axis 3 in the 7-D NIfTI array)
        arr_dec = arr[:, :, :, ::factor, ...]

        new_dt = old_dt * factor
        cf = getattr(hdr, 'SpectrometerFrequency', [123.259])[0]

        # Rebuild dim tags from original
        dim_tags = [None, None, None]
        for i, dim_idx in enumerate([5, 6, 7]):
            tag = getattr(hdr, f'dim_{dim_idx}', None)
            if tag is not None and tag != 'N/A':
                dim_tags[i] = tag

        new_obj = gen_nifti_mrs(
            data=arr_dec,
            dwelltime=new_dt,
            spec_freq=cf,
            nucleus='1H',
            dim_tags=dim_tags,
            no_conj=True,
        )
        return new_obj

    def load_mats(self):
        data, refs, MM_data, MM_refs, names = [], [], [], [], []
        self.load_failures = []  # (filename, error_msg)

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
                                    try:
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
                                    except Exception as e:
                                        self.load_failures.append((file, str(e)))
                                        print(f"Error loading {file}: {e}")

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
        print(elem.hdr_ext)
        elem.plot()
        plt.show()