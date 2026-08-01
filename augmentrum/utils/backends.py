####################################################################################################
#                                          backends.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-07-31                                                                              #
#                                                                                                  #
# Purpose: Array-constructor backends. Defines the ArrayBackend contract that every backend must   #
#          satisfy, plus the NumPy and PyTorch implementations of it.                              #
#          Not to be confused with augmentrum.core.Backend, which is an enum naming a storage      #
#          format; this file is about which framework new arrays are created in.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math
from abc import ABC, abstractmethod

import numpy as np


#**************************************************************************************************#
#                                        Class ArrayBackend                                        #
#**************************************************************************************************#
#                                                                                                  #
# Contract for array-constructor backends.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class ArrayBackend(ABC):
    """
    Contract for array-constructor backends.

    Subclasses must implement every method below. Because these are true
    abstract methods, an incomplete backend raises ``TypeError`` the moment it
    is constructed, naming the missing operations — the previous docstring-only
    interface deferred that failure until a generator happened to call the
    missing method.

    Two invariants the trajectory generators rely on:

    * ``linspace`` takes ``endpoint`` as its fourth **positional** parameter.
    * every backend honours the ``dtype`` given at construction, so a
      ``NumpyBackend`` and a ``TorchBackend`` built with the same dtype are
      interchangeable.

    >>> backend = NumpyBackend()
    >>> backend.shape(backend.linspace(-32.0, 31.0, 64))
    (64,)
    """

    #******************#
    #   construction   #
    #******************#

    @abstractmethod
    def __init__(self, device: str = "cpu", dtype=None):
        """Backends are constructed with a device and a floating dtype."""

    #********************#
    #   array creation   #
    #********************#

    @abstractmethod
    def array(self, data, dtype=None):
        """Create a backend array from a Python list or numpy array."""

    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Array of zeros."""

    @abstractmethod
    def ones(self, shape, dtype=None):
        """Array of ones."""

    @abstractmethod
    def full(self, shape, fill_value, dtype=None):
        """Array filled with *fill_value*."""

    @abstractmethod
    def linspace(self, start, stop, num, endpoint=True):
        """Evenly-spaced values, like ``numpy.linspace``."""

    @abstractmethod
    def arange(self, start, stop=None, step=1):
        """Evenly-spaced values, like ``numpy.arange``."""

    @abstractmethod
    def meshgrid(self, *arrays, indexing="ij"):
        """N-D coordinate grids."""

    #*****************#
    #   combination   #
    #*****************#

    @abstractmethod
    def stack(self, arrays, axis=0):
        """Stack a sequence of arrays along a new axis."""

    @abstractmethod
    def concat(self, arrays, axis=0):
        """Concatenate arrays along an existing axis."""

    #**********#
    #   math   #
    #**********#

    @abstractmethod
    def sin(self, x):
        """Element-wise sine."""

    @abstractmethod
    def cos(self, x):
        """Element-wise cosine."""

    @abstractmethod
    def sqrt(self, x):
        """Element-wise square root."""

    @abstractmethod
    def floor(self, x):
        """Element-wise floor."""

    @abstractmethod
    def ceil(self, x):
        """Element-wise ceiling."""

    #****************#
    #   conversion   #
    #****************#

    @abstractmethod
    def asarray(self, x):
        """Convert to a backend array; no-op when already the right type."""

    @abstractmethod
    def shape(self, x):
        """Shape as a plain Python tuple."""

    @abstractmethod
    def to_cpu(self, x):
        """Return a numpy view or copy, for I/O and assertions."""

    @abstractmethod
    def cast_bool(self, x):
        """Cast to boolean."""

    @abstractmethod
    def cast_int(self, x):
        """Cast to 64-bit integer."""

    @abstractmethod
    def cast_float(self, x):
        """Cast to the backend's floating dtype."""

    #************#
    #   random   #
    #************#

    @abstractmethod
    def randperm(self, n):
        """Random permutation of ``range(n)`` as a 1-D integer array."""

    #***************#
    #   constants   #
    #***************#

    def pi(self) -> float:
        """
        Pi as a Python float.

        Concrete, not abstract: it is the same value in every framework, so
        requiring each backend to restate it bought nothing.
        """
        return math.pi


#**************************************************************************************************#
#                                        Class NumpyBackend                                        #
#**************************************************************************************************#
#                                                                                                  #
# Array backend backed by NumPy.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
class NumpyBackend(ArrayBackend):
    """
    Array backend backed by NumPy.

    The *device* argument exists only to satisfy the shared contract; NumPy has
    no device concept and anything other than ``'cpu'`` is rejected rather than
    silently ignored.
    """

    def __init__(self, device: str = "cpu", dtype=None):
        if device != "cpu":
            raise ValueError(f"NumpyBackend supports device='cpu' only, got {device!r}.")
        self.device = "cpu"
        self._dtype = dtype or np.float32

    #********************#
    #   array creation   #
    #********************#

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or self._dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or self._dtype)

    def full(self, shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype or self._dtype)

    def linspace(self, start, stop, num, endpoint=True):
        return np.linspace(start, stop, int(num), dtype=self._dtype, endpoint=endpoint)

    def arange(self, start, stop=None, step=1):
        if stop is None:
            return np.arange(start, dtype=self._dtype)
        return np.arange(start, stop, step, dtype=self._dtype)

    def meshgrid(self, *arrays, indexing="ij"):
        return np.meshgrid(*arrays, indexing=indexing)

    #*****************#
    #   combination   #
    #*****************#

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def concat(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    #**********#
    #   math   #
    #**********#

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def floor(self, x):
        return np.floor(x)

    def ceil(self, x):
        return np.ceil(x)

    #****************#
    #   conversion   #
    #****************#

    def asarray(self, x):
        return np.asarray(x, dtype=self._dtype)

    def shape(self, x):
        return tuple(x.shape)

    def to_cpu(self, x):
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    def cast_bool(self, x):
        return x.astype(bool)

    def cast_int(self, x):
        return x.astype(np.int64)

    def cast_float(self, x):
        return x.astype(self._dtype)

    #************#
    #   random   #
    #************#

    def randperm(self, n):
        return np.random.permutation(n).astype(np.int64)

    def __repr__(self):
        return f"NumpyBackend(dtype={np.dtype(self._dtype).name})"


#**************************************************************************************************#
#                                        Class TorchBackend                                        #
#**************************************************************************************************#
#                                                                                                  #
# Array backend backed by PyTorch.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class TorchBackend(ArrayBackend):
    """
    Array backend backed by PyTorch.

    >>> backend = TorchBackend(device='cuda')       # doctest: +SKIP
    >>> kx = backend.linspace(-32.0, 31.0, 64)      # doctest: +SKIP
    """

    def __init__(self, device: str = "cpu", dtype=None):
        import torch  # deferred so a numpy-only environment can import this module
        self._torch = torch
        self.device = torch.device(device)
        self._dtype = dtype or torch.float32

    #********************#
    #   array creation   #
    #********************#

    def array(self, data, dtype=None):
        return self._torch.tensor(data, dtype=dtype or self._dtype, device=self.device)

    def zeros(self, shape, dtype=None):
        return self._torch.zeros(shape, dtype=dtype or self._dtype, device=self.device)

    def ones(self, shape, dtype=None):
        return self._torch.ones(shape, dtype=dtype or self._dtype, device=self.device)

    def full(self, shape, fill_value, dtype=None):
        return self._torch.full(shape, fill_value, dtype=dtype or self._dtype,
                                device=self.device)

    def linspace(self, start, stop, num, endpoint=True):
        # Signature must match NumpyBackend.linspace: CartesianTrajectory passes
        # `endpoint` positionally. torch.linspace always includes the stop value,
        # so endpoint=False is emulated by generating one extra point and dropping it.
        num = int(num)
        if endpoint:
            return self._torch.linspace(start, stop, num, dtype=self._dtype,
                                        device=self.device)
        return self._torch.linspace(start, stop, num + 1, dtype=self._dtype,
                                    device=self.device)[:-1]

    def arange(self, start, stop=None, step=1):
        if stop is None:
            return self._torch.arange(start, dtype=self._dtype, device=self.device)
        return self._torch.arange(start, stop, step, dtype=self._dtype, device=self.device)

    def meshgrid(self, *arrays, indexing="ij"):
        return self._torch.meshgrid(*arrays, indexing=indexing)

    #*****************#
    #   combination   #
    #*****************#

    def stack(self, arrays, axis=0):
        return self._torch.stack(arrays, dim=axis)

    def concat(self, arrays, axis=0):
        return self._torch.cat(arrays, dim=axis)

    #**********#
    #   math   #
    #**********#

    def sin(self, x):
        return self._torch.sin(x)

    def cos(self, x):
        return self._torch.cos(x)

    def sqrt(self, x):
        return self._torch.sqrt(x)

    def floor(self, x):
        return self._torch.floor(x)

    def ceil(self, x):
        return self._torch.ceil(x)

    #****************#
    #   conversion   #
    #****************#

    def asarray(self, x):
        if isinstance(x, self._torch.Tensor):
            return x.to(dtype=self._dtype, device=self.device)
        return self._torch.tensor(x, dtype=self._dtype, device=self.device)

    def shape(self, x):
        return tuple(x.shape)

    def to_cpu(self, x):
        if isinstance(x, self._torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def cast_bool(self, x):
        return x.bool()

    def cast_int(self, x):
        return x.long()

    def cast_float(self, x):
        return x.to(self._dtype)

    #************#
    #   random   #
    #************#

    def randperm(self, n):
        return self._torch.randperm(n, device=self.device)

    def __repr__(self):
        return f"TorchBackend(device={self.device}, dtype={self._dtype})"


#************************#
#   backend validation   #
#************************#
def validate_backend(backend) -> None:
    """
    Raise unless *backend* is a usable :class:`ArrayBackend`.

    This is now an ``isinstance`` check. It used to re-list all 22 method names
    by hand, a second copy of the contract that nothing kept in sync with the
    interface it was supposed to mirror.
    """
    if not isinstance(backend, ArrayBackend):
        raise TypeError(
            f"Backend {backend!r} is not an ArrayBackend. Subclass "
            "augmentrum.utils.backends.ArrayBackend and implement its abstract methods."
        )
