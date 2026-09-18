####################################################################################################
#                                          precision.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-18                                                                              #
#                                                                                                  #
# Purpose: The working precision of a module: 'single' (float32 / complex64) or 'double'           #
#          (float64 / complex128), or None to follow the data it is given. Every module takes it   #
#          as "precision"; the data are cast to it on the way in, and every computation inside    #
#          follows the dtype of the data it works on, so the output is in that precision too.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Optional

from nifti_mrs_plus import ops


__all__ = ['PRECISIONS', 'check', 'of', 'dtype_name', 'cast']


#: The precisions a module can be asked to work in.
PRECISIONS = ('single', 'double')

_REAL = {'single': 'float32', 'double': 'float64'}
_COMPLEX = {'single': 'complex64', 'double': 'complex128'}
_OF = {'float16': 'single', 'bfloat16': 'single', 'float32': 'single', 'complex64': 'single',
       'float64': 'double', 'complex128': 'double'}


#***********#
#   check   #
#***********#
def check(precision: Optional[str]) -> Optional[str]:
    """*precision* if it is one of "PRECISIONS" or None, else a ValueError."""
    if precision is not None and precision not in PRECISIONS:
        raise ValueError(f"precision must be None (follow the data) or one of {PRECISIONS}, "
                         f"got {precision!r}")
    return precision


#********#
#   of   #
#********#
def _name(x) -> str:
    """The dtype of *x* as a plain name ('complex64'), whatever the backend."""
    dtype = getattr(x, 'dtype', None)
    if dtype is None:
        return ''
    name = getattr(dtype, 'name', None)                  # NumPy, TensorFlow
    return name if isinstance(name, str) else str(dtype).rpartition('.')[2]   # torch.complex64


def of(x) -> Optional[str]:
    """'single' or 'double' for floating or complex *x*, None for anything else."""
    return _OF.get(_name(x))


#****************#
#   dtype name   #
#****************#
def dtype_name(x, precision: Optional[str] = None) -> str:
    """
    The dtype name *x* has in *precision*, keeping it real or complex: complex64 in 'double'
    is complex128. None keeps *x*'s own precision; integer and boolean dtypes are kept.
    """
    name = _name(x)
    target = precision or _OF.get(name)
    if target is None or name not in _OF:
        return name
    return (_COMPLEX if name.startswith('complex') else _REAL)[target]


#**********#
#   cast   #
#**********#
def cast(x, precision: Optional[str]):
    """
    *x* in *precision*; unchanged (the same object) when *precision* is None, when *x* is None
    or not floating, and when it is already in that precision.
    """
    if precision is None or x is None:
        return x
    have = of(x)
    if have is None or have == precision:
        return x
    return ops.cast(x, dtype_name(x, precision))
