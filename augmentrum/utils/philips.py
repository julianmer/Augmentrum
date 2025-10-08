####################################################################################################
#                                            loadData.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 20/06/22                                                                                #
#                                                                                                  #
# Purpose: Contains various Philips specific utils.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np


#****************#
#   definition   #
#****************#
index_options = ['STD', 'REJ', 'PHC', 'FRC', 'NOI', 'NAV']

index_headers = ['typ', 'mix', 'dyn', 'card', 'echo', 'loca', 'chan',
                 'extr1', 'extr2', 'kx', 'ky', 'kz', 'aver', 'sign',
                 'rf', 'grad', 'enc', 'rtop', 'rr', 'size', 'offset']

indices = ['chan', 'aver', 'dyn', 'card', 'echo', 'loca', 'extr1', 'extr2']

defaults = {'chan': 'DIM_COIL',
            'aver': 'DIM_DYN',
            'dyn': 'DIM_DYN',
            'echo': 'DIM_INDIRECT_0'}


#***********************#
#   read a .data file   #
#***********************#
def read_Philips_data(data_file, df):
    """
    Reads a .data file with given .list data frame. Parts of the code are taken from:
    https://github.com/wtclarke/spec2nii/blob/master/spec2nii/philips_data_list.py

    @param data_file -- The path to the data file.
    @param df -- The data frame obtained from the .list file.

    @returns -- A dict containing the data.
    """
    with open(data_file, 'rb') as f:
        raw = np.fromfile(f, dtype='<f4')
        raw = raw[0::2] + 1j * raw[1::2]

    data_types = df.typ.unique()
    output_dict = {}
    for tt in data_types:
        curr_df = df.loc[df.typ == tt, :]
        if tt == 'NOI':
            # special simple case for NOI
            spec_res = int(curr_df['size'].max() / 8)
            ncha = curr_df['chan'].max() + 1
            nloca = curr_df['loca'].max() + 1
            output_dict[tt] = np.zeros((spec_res, ncha, nloca), dtype=np.complex128)
        else:
            n_mix = curr_df['mix'].max() + 1
            # other types might use all the loops
            for mix in range(n_mix):
                curr_mix_df = curr_df.loc[curr_df.mix == mix, :]
                shape = []
                shape.append(int(curr_mix_df['size'].max() / 8))
                for ind in indices:
                    shape.append(curr_mix_df[ind].max() + 1)
                output_dict[f'{tt}_{mix}'] = np.zeros(shape, dtype=np.complex128)

    # now extract data
    offset = 0
    for ind in range(df.shape[0]):
        cdf = df.iloc[ind]
        tt = cdf.typ
        dsize = int(cdf['size'].max() / 8)
        if tt == 'NOI':
            output_dict[tt][:, cdf.chan, cdf.loca] = raw[offset:(offset + dsize)]
        else:
            mix = cdf.mix
            ind = [cdf[ii] for ii in indices]
            ind = tuple([slice(None), ] + ind)
            output_dict[f'{tt}_{mix}'][ind] = raw[offset:(offset + dsize)]
        offset += dsize

    return output_dict