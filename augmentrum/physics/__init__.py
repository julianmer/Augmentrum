####################################################################################################
#                                        physics/__init__.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. T. LaMaster (john.t.lamaster@gmail.com)                                              #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: Re-exports the ".seq" file reader that supplies a real gradient waveform/trajectory -    #
#          see "augmentrum.sampling.kspace_sampling.SeqFile" for where it plugs into the pipeline.  #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
from augmentrum.physics.seq_file import SeqFileData, load_seq_file, GAMMA_HZ_PER_T

__all__ = ['SeqFileData', 'load_seq_file', 'GAMMA_HZ_PER_T']
