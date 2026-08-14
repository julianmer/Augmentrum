####################################################################################################
#                                          biggaba.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-14                                                                              #
#                                                                                                  #
# Purpose: Loads the Big GABA repository (Mikkelsen et al.) - multi-site, multi-vendor GABA-edited #
#          MEGA-PRESS - into NIfTI-MRS with a proper DIM_EDIT axis, for edited augmentation work.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import pathlib

import numpy as np

# own
from augmentrum import Augmentrum


#*************************#
#   biggaba data loader   #
#*************************#
def BigGABAData(data_dir, te='68', sites=None, limit=None, batch_size=16, seed=0,
                val_frac=0.1, test_frac=0.1, pipelines=None, modes=None,
                backend='pytorch', volatile=False, **kwargs):
    """
    Load Big GABA MEGA-PRESS data and create an Augmentrum instance.

    Args:
        data_dir: Path to the Big GABA root (holding "MEGA_PRESS/") or to the
            "MEGA_PRESS" directory itself.
        te: Which acquisition to keep, by the TE token in the file names —
            '68' (GABA, default) or '80'.
        sites: Iterable of site prefixes to keep, e.g. ("P5", "S5"). None
            keeps every site.
        limit: At most this many scans per site directory. None keeps all.
        batch_size: Batch size for dataloaders.
        seed: Random seed for the split.
        val_frac: Validation fraction (default 0.1).
        test_frac: Test fraction (default 0.1).
        pipelines: Custom pipelines dict or None for defaults.
        modes: Sampling modes dict or None for defaults.
        backend: Backend to use ('numpy', 'pytorch', etc.).
        volatile: If True, skip provenance logging.
        **kwargs: Additional parameters for modules.

    Returns:
        Augmentrum instance with Big GABA data loaded.
    """
    if modes is None:
        modes = {'train': 'random', 'val': 'deterministic', 'test': 'deterministic'}

    loader = BigGABAModule(data_dir, te=te, sites=sites, limit=limit)
    data, water, names = loader.load()

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
        **kwargs
    )


#**************************************************************************************************#
#                                       Class BigGABAModule                                        #
#**************************************************************************************************#
#                                                                                                  #
# Walks the Big GABA tree and loads each scan as NIfTI-MRS with a DIM_EDIT axis.                   #
#                                                                                                  #
#**************************************************************************************************#
class BigGABAModule:
    """
    Walks the Big GABA tree and loads each scan as NIfTI-MRS with a DIM_EDIT axis.

    One repository, three vendors, three raw formats - and three different
    places the edit loop hides:

    - GE P-files ("*.7"): spec2nii's GABA mapper reads them complete with
      DIM_EDIT and an embedded water reference. Nothing to do.
    - Siemens TWIX ("*.dat" + "*_H2O.dat"): the edit loop arrives as a
      length-2 DIM_USER_0, so it is retagged DIM_EDIT; the 2x readout
      oversampling is removed.
    - Philips SDAT/SPAR ("*_act" + "*_ref"): the 320 rows interleave the two
      edit conditions row by row, with the second condition stored
      phase-inverted (verified on the data: the parity split alone shows the
      edited GABA 3.0 / Glx 3.7 ppm signature after undoing the sign, and
      the parity groups differ in magnitude, which a pure phase cycle
      cannot). Reshaped to "(dyn, edit)"; the inversion is left in the data
      and recorded here rather than silently corrected.

    Dimension order differs per vendor; the dim tags carry the meaning, so
    downstream modules never need the order to match across subjects.

    Args:
        data_dir: Big GABA root or its "MEGA_PRESS" directory.
        te: TE token to keep, '68' (default) or '80'.
        sites: Site prefixes to keep, e.g. ("P5",). None keeps all.
        limit: At most this many scans per site directory. None keeps all.
    """

    def __init__(self, data_dir, te: str = '68', sites=None, limit=None):
        root = pathlib.Path(data_dir)
        if (root / 'MEGA_PRESS').is_dir():
            root = root / 'MEGA_PRESS'
        self.root = root
        self.te = str(te)
        self.sites = None if sites is None else tuple(sites)
        self.limit = limit

        self.load_failures = []  # (path, error message)

    #*************#
    #   walking   #
    #*************#
    def load(self):
        """
        Load every matching scan.

        Returns:
            "(data, water, names)" - lists of NIFTI_MRS objects (water entries
            may be None where a scan has no reference) and scan names.
        """
        data, water, names = [], [], []

        for site_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            site = site_dir.name.split('_')[0]
            if self.sites is not None and site not in self.sites:
                continue

            scans = sorted(p for p in site_dir.iterdir() if p.is_dir())
            for scan_dir in (scans if self.limit is None else scans[:self.limit]):
                try:
                    loaded = self._load_scan(scan_dir)
                except Exception as error:
                    self.load_failures.append((str(scan_dir), str(error)))
                    print(f"Error loading {scan_dir}: {error}")
                    continue
                if loaded is not None:
                    met, ref = loaded
                    data.append(met)
                    water.append(ref)
                    names.append(f"{site}_{scan_dir.name}_TE{self.te}")

        return data, water, names

    def _load_scan(self, scan_dir: pathlib.Path):
        """One scan directory, dispatched by the raw format found inside."""
        files = sorted(os.listdir(scan_dir))
        token = f"_{self.te}"

        pfiles = [f for f in files if f.endswith('.7') and token in f]
        if pfiles:
            return self.load_ge(scan_dir / pfiles[0])

        twix = [f for f in files if f.endswith('.dat') and token in f
                and '_H2O' not in f]
        if twix:
            met = scan_dir / twix[0]
            ref = scan_dir / twix[0].replace('.dat', '_H2O.dat')
            return self.load_siemens(met, ref if ref.exists() else None)

        sdat = [f for f in files if f.endswith('_act.SDAT') and token in f]
        if sdat:
            met = scan_dir / sdat[0]
            ref = scan_dir / sdat[0].replace('_act.SDAT', '_ref.SDAT')
            return self.load_philips(met, ref if ref.exists() else None)

        return None

    #**********#
    #   ge     #
    #**********#
    @staticmethod
    def load_ge(pfile_path: pathlib.Path):
        """GE P-file: metabolite and embedded reference, DIM_EDIT included."""
        from spec2nii.GE.ge_pfile import read_pfile

        images, names = read_pfile(pfile_path, pfile_path.stem)

        met, ref = None, None
        for image, name in zip(images, names):
            if name.endswith('_ref'):
                ref = image
            else:
                met = image
        return met, ref

    #*************#
    #   siemens   #
    #*************#
    def load_siemens(self, dat_path: pathlib.Path, h2o_path=None):
        """Siemens TWIX: retag the edit loop, drop the 2x oversampling."""
        met = self._read_twix(dat_path)
        ref = self._read_twix(h2o_path) if h2o_path is not None else None
        return met, ref

    @classmethod
    def _read_twix(cls, path: pathlib.Path):
        from mapvbvd import mapVBVD
        from spec2nii.Siemens.twixfunctions import process_twix

        twix = mapVBVD(str(path), quiet=True)
        if isinstance(twix, list):
            twix = twix[-1]

        overrides = {'dims': (None, None, None), 'tags': (None, None, None)}
        images, _ = process_twix(twix, path.stem, path.name, 'image',
                                 overrides, quiet=True)
        nifti = images[0]

        # The edit loop is the length-2 DIM_USER_0 the generic reader emits.
        for position, tag in enumerate(nifti.dim_tags):
            if tag == 'DIM_USER_0':
                nifti.set_dim_tag(position + 4, 'DIM_EDIT')

        return cls._decimated(nifti)

    @staticmethod
    def _decimated(nifti, factor: int = 2):
        """
        Remove the Siemens 2x readout oversampling.

        Every other point along the spectral axis, and the dwell time doubled
        to match - the acquisition the protocol actually specifies.
        """
        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

        array = np.asarray(nifti[:])[:, :, :, ::factor, ...]
        tags = list(nifti.dim_tags)

        out = gen_nifti_mrs(
            data=array,
            dwelltime=float(nifti.dwelltime) * factor,
            spec_freq=nifti.spectrometer_frequency[0],
            nucleus=nifti.nucleus[0] if nifti.nucleus else '1H',
            dim_tags=tags + [None] * (3 - len(tags)),
        )
        return out

    #*************#
    #   philips   #
    #*************#
    @staticmethod
    def load_philips(act_path: pathlib.Path, ref_path=None):
        """Philips SDAT: rows interleave the edit conditions pairwise."""
        met = BigGABAModule._read_sdat(act_path)
        ref = BigGABAModule._read_sdat(ref_path) if ref_path is not None else None
        return met, ref

    @staticmethod
    def _read_sdat(path: pathlib.Path):
        from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
        from spec2nii.Philips.philips import read_sdat, read_spar

        params = read_spar(str(path)[:-4] + 'SPAR')
        rows = int(params['rows'])
        data = read_sdat(str(path), params['samples'], rows)

        # (T, rows) -> (T, dyn, edit): the conditions interleave row by row,
        # so the edit index is the fast axis of each row pair. Note the
        # second condition is stored phase-inverted (Philips convention).
        array = data.reshape(data.shape[0], rows // 2, 2)

        nifti = gen_nifti_mrs(
            data=array[None, None, None],
            dwelltime=1.0 / float(params['sample_frequency']),
            spec_freq=float(params['synthesizer_frequency']) / 1e6,
            nucleus='1H',
            dim_tags=['DIM_DYN', 'DIM_EDIT', None],
        )
        nifti.add_hdr_field('EchoTime', float(params['echo_time']) / 1000.0)
        nifti.add_hdr_field('RepetitionTime', float(params['repetition_time']) / 1000.0)
        return nifti
