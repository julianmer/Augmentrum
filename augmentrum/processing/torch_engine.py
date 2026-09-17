####################################################################################################
#                                        torch_engine.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-17                                                                              #
#                                                                                                  #
# Purpose: The batched device engine of RawProcessor. Every estimate of the FSL-MRS raw pipeline   #
#          - coil weights, spectral registration, unlike-transient detection, eddy current phase,  #
#          reference peaks - as whole-batch torch operations on the data's own device, with        #
#          per-sample coil and transient masks standing in for ragged subsets - and the replay of  #
#          such a computation as CUDA graphs.                                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import functools
import math
from collections import OrderedDict

import numpy as np
import torch

# own
from augmentrum.processing.utils import ppm_shift_axis, ppm_window


__all__ = ['fid_to_spec', 'masked_median', 'noise_moments', 'noise_covariance', 'reference_gram',
           'combine_coils', 'principal_vector', 'wsvd_weights', 'align', 'alignment_phasor',
           'unlike_mask', 'unwrap', 'ecc_phase', 'peak_phase', 'peak_shift_hz', 'shift_phasor',
           'first_true', 'upload', 'constant', 'window_spans', 'Step', 'run_steps', 'GraphedSteps',
           'wsvd_weight_steps', 'peak_shift_steps']


#: FSL-MRS estimate_noise_cov: the last tenth of every FID is noise.
NOISE_FRACTION = 0.1

#: FSL-MRS estimate_noise_cov refuses a covariance from fewer samples per coil than this.
MIN_SAMPLES_PER_COIL = 10


#********************#
#   fsl primitives   #
#********************#
def halve_first(fids):
    """The FIDs with their first point halved, as FSL-MRS FIDToSpec takes them."""
    return torch.cat([fids[..., :1] * 0.5, fids[..., 1:]], dim=-1)


def fid_to_spec(fids):
    """FSL-MRS FIDToSpec along the last axis: first point halved, ortho fft, fftshift."""
    return torch.fft.fftshift(torch.fft.fft(halve_first(fids), dim=-1, norm='ortho'), dim=-1)


def first_true(mask):
    """Index of the first True along the last axis (0 where there is none)."""
    return mask.to(torch.int8).argmax(dim=-1)


#**********************#
#   device transfers   #
#**********************#
# Every upload of a host array to an accelerator is a synchronisation point:
# the host waits for all the work queued before it. Constants are uploaded once
# per device and kept; per-batch values go through pinned memory without waiting.
_CONSTANTS = {}

#: Lists collecting every constant handed out while a graph is captured (see "GraphedSteps").
_RECORDERS = []


def upload(array, device, dtype=None):
    """*array* on *device*, without waiting for the device where it can be avoided."""
    tensor = torch.as_tensor(np.asarray(array), dtype=dtype)
    if torch.device(device).type != 'cuda':
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def constant(key, build, device):
    """A host-built constant, uploaded to *device* once and kept (bounded)."""
    key = (key, str(device))
    if key not in _CONSTANTS:
        if len(_CONSTANTS) > 256:
            _CONSTANTS.clear()
        _CONSTANTS[key] = upload(build(), device)
    for recorder in _RECORDERS:
        recorder.append(_CONSTANTS[key])
    return _CONSTANTS[key]


@functools.lru_cache(maxsize=4096)
def window_spans(n, sw_hz, sf_mhz, ppmlim):
    """"ppm_window" for a hashable *ppmlim*, remembered: the same few windows every batch."""
    return ppm_window(n, sw_hz, sf_mhz, ppmlim)


#****************************#
#   stepwise computations    #
#****************************#
# A batched estimate launches thousands of small kernels, and on a GPU the host
# spends far longer launching them than the device running them. A CUDA graph
# records the kernels once and relaunches them in one call, with the very same
# arithmetic - as long as nothing in between needs the host. The few things that
# do are handed out as Steps by computations written as generators, which run
# either eagerly ("run_steps") or as graphs around the steps ("GraphedSteps").
class Step:
    """
    What a stepwise computation cannot do inside a CUDA graph, handed out to run eagerly.

    A stepwise computation is a generator. Where it needs an operation a graph
    cannot hold - a library call that allocates device memory of its own, or
    arithmetic on a Python value that changes from call to call and would be
    baked into a graph - it yields a Step and is sent the result back. Such
    values never reach the computation itself: a Step names them in *late*,
    and whoever runs it supplies them.

    Args:
        fn: The operation.
        *args: Its positional arguments.
        late: Names of keyword arguments supplied when it runs.
        **kwargs: Its fixed keyword arguments.
    """

    __slots__ = ('fn', 'args', 'kwargs', 'late')

    def __init__(self, fn, *args, late=(), **kwargs):
        self.fn, self.args, self.kwargs, self.late = fn, args, kwargs, tuple(late)

    def run(self, values=None):
        """The operation's result, *values* supplying the late arguments."""
        extra = {name: (values or {})[name] for name in self.late}
        return self.fn(*self.args, **self.kwargs, **extra)


def run_steps(steps, values=None):
    """
    Run a stepwise computation eagerly, start to end.

    Args:
        steps: The computation's generator.
        values: The late values its steps name.

    Returns:
        What the computation returns.
    """
    try:
        request = next(steps)
        while True:
            request = steps.send(request.run(values))
    except StopIteration as done:
        return done.value


def _copied(value):
    """*value* with every tensor cloned and every container rebuilt, other leaves shared."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_copied(v) for v in value)
    if isinstance(value, dict):
        return {k: _copied(v) for k, v in value.items()}
    if isinstance(value, set):
        return set(value)
    return value


def _copy_into(target, value):
    """Write *value*'s tensors into *target*'s, a structure of the same shape."""
    if torch.is_tensor(target):
        target.copy_(value)
    elif isinstance(target, (list, tuple)):
        for t, v in zip(target, value):
            _copy_into(t, v)
    elif isinstance(target, dict):
        for k in target:
            _copy_into(target[k], value[k])


class GraphedSteps:
    """
    Stepwise computations replayed as CUDA graphs, one set per signature.

    Recording needs everything that decides which kernels run to be fixed -
    shapes, flags, the Python values a computation reads - so a caller states
    all of it as a *signature*. The tensors that vary from call to call are
    the *inputs*: a replay copies them into the graphs' own, runs the graph up
    to the first step, the step, the next graph, and so on. A signature runs
    eagerly for its first *warmup* calls, is recorded on the next, and is
    replayed from then on; the *capacity* most recently used are kept.

    A replay is the recorded kernels on the recorded tensors, so its results
    are the eager ones bit for bit. They are handed out as copies, which
    outlive the next replay.
    """

    def __init__(self, warmup=2, capacity=4):
        self.warmup = int(warmup)
        self.capacity = int(capacity)
        self._entries = OrderedDict()

    def __call__(self, signature, steps, inputs, values=None, keep=()):
        """
        Run a computation, as graphs where they are recorded.

        Args:
            signature: Hashable; everything besides *inputs* and *values*
                the computation depends on.
            steps: Callable taking *inputs* and returning a fresh generator
                of the computation.
            inputs: {name: tensor or None}; a replay expects the same names,
                and tensors of the same shape, strides, dtype and device.
            values: The late values of the computation's steps.
            keep: What the computation reads besides its inputs - cached
                tensors - kept alive for as long as the graphs, which read
                its memory.

        Returns:
            The computation's result; its tensors are the caller's.
        """
        entry = self._entries.get(signature)
        if entry is None:
            while len(self._entries) >= max(self.capacity, 1):
                self._entries.popitem(last=False)
            entry = self._entries[signature] = _GraphEntry()
        self._entries.move_to_end(signature)
        if entry.graphs is None:
            if entry.calls < self.warmup:
                entry.calls += 1
                return run_steps(steps(inputs), values)
            entry.record(steps, inputs, values, keep)
        return entry.replay(inputs, values)

    def clear(self):
        """Forget every recorded graph, and the memory it holds."""
        self._entries.clear()

    def __len__(self):
        return sum(entry.graphs is not None for entry in self._entries.values())

    def __getstate__(self):
        # graphs belong to a process and a device: a copy starts without any
        return {'warmup': self.warmup, 'capacity': self.capacity}

    def __setstate__(self, state):
        self.__init__(**state)


class _GraphEntry:
    """One signature: its inputs, its graphs and the steps between them."""

    def __init__(self):
        self.calls = 0
        self.graphs = None

    def record(self, steps, inputs, values, keep):
        """
        Record the computation on the graphs' own copies of *inputs*.

        A run on a side stream comes first, as torch.cuda.graph asks, so that
        library handles and workspaces exist before recording. The graphs
        share one memory pool, which is safe because they are replayed in the
        order they were recorded. Constants the computation fetches while it
        is recorded are kept with the graphs.
        """
        self.inputs = {name: (tensor.clone() if tensor is not None else None)
                       for name, tensor in inputs.items()}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run_steps(steps(self.inputs), values)
        torch.cuda.current_stream().wait_stream(side)

        pool = torch.cuda.graph_pool_handle()
        graphs, requests, sent, constants = [], [], [], []
        _RECORDERS.append(constants)
        try:
            run, value = steps(self.inputs), None
            while True:
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                try:
                    with torch.cuda.graph(graph, pool=pool):
                        request = run.send(value)
                except StopIteration as done:
                    result = done.value
                    break
                requests.append(request)
                # the step's result, in memory of its own: the next graph's input
                value = _copied(request.run(values))
                sent.append(value)
        finally:
            _RECORDERS.remove(constants)
        self.result, self.requests, self.sent = result, requests, sent
        self.held = (keep, constants, pool)
        self.graphs = graphs

    def replay(self, inputs, values):
        """The computation on *inputs*: copy them in, then graph, step, graph, ..."""
        for name, tensor in inputs.items():
            if tensor is not None:
                self.inputs[name].copy_(tensor)
        for index, graph in enumerate(self.graphs):
            graph.replay()
            if index < len(self.requests):
                _copy_into(self.sent[index], self.requests[index].run(values))
        return _copied(self.result)


#***********************#
#   masked statistics   #
#***********************#
def masked_median(x, mask):
    """
    Median over the next-to-last axis of complex *x*, real and imaginary apart.

    NumPy's convention - the mean of the two middle values for an even count -
    which torch.median does not follow (it returns the lower one). Entries
    outside *mask* are sorted behind every valid one and never reached.

    Args:
        x: Complex tensor, (..., D, T).
        mask: Boolean keep mask, (..., D), at least one True per row.

    Returns:
        The complex median, (..., T).
    """
    count = mask.sum(dim=-1)
    low = ((count - 1) // 2)[..., None, None].expand(*x.shape[:-2], 1, x.shape[-1])
    high = (count // 2)[..., None, None].expand(*x.shape[:-2], 1, x.shape[-1])
    keep = mask[..., None]

    def median(values):
        ordered = torch.where(keep, values, torch.inf).sort(dim=-2).values
        return 0.5 * (ordered.gather(-2, low) + ordered.gather(-2, high))[..., 0, :]

    return torch.complex(median(x.real), median(x.imag))


#**********************#
#   coil combination   #
#**********************#
def noise_moments(tails):
    """
    First and second moments of the noise tails, one set per transient.

    The coil covariance of any subset of transients is a weighted sum of these,
    and that of any subset of coils is a sub-matrix of it, so they are what a
    pool of scans can cache once for every draw.

    Args:
        tails: Noise samples, (B, V, C, D, L) complex - voxels V, coils C,
            transients D, and the last L points of every FID.

    Returns:
        "(second, first)": sum of x x^H, (B, D, C, C), and sum of x, (B, D, C),
        both over voxels and points, in complex128.
    """
    b, v, c, d, l = tails.shape
    x = tails.to(torch.complex128).permute(0, 3, 1, 4, 2).reshape(b, d, v * l, c)
    return x.mT @ x.conj(), x.sum(dim=-2)


def noise_covariance(second, first, samples_per_transient, dyn_mask=None):
    """
    The per-subject coil covariance of np.cov, from cached moments.

    Args:
        second: Sum of x x^H per transient, (B, D, C, C).
        first: Sum of x per transient, (B, D, C).
        samples_per_transient: Noise samples one transient contributes.
        dyn_mask: Transients to pool, (B, D) bool; None pools them all.

    Returns:
        "(cov, n)": the covariance (B, C, C) and the number of samples (B,).
    """
    if dyn_mask is None:
        dyn_mask = torch.ones(second.shape[:2], dtype=torch.bool, device=second.device)
    weights = dyn_mask.to(torch.float64)
    total = (second * weights[..., None, None]).sum(dim=1)
    mean_sum = (first * weights[..., None]).sum(dim=1)
    n = weights.sum(dim=1) * samples_per_transient
    outer = mean_sum[:, :, None] * mean_sum[:, None, :].conj()
    cov = (total - outer / n[:, None, None]) / (n - 1).clamp(min=1)[:, None, None]
    return cov, n


def reference_gram(reference):
    """X^H X of reference FIDs laid out (..., T, C): the right-singular problem of wSVD."""
    x = reference.to(torch.complex128)
    return x.mH @ x


def combine_coils(x, weights):
    """
    "sum_c x[:, :, c] w[..., c]" for x (B, V, C, D, T) and weights (B, V, C) or (B, V, D, C).

    A batch drawn from NIfTI data is stored transients-last, (B, V, T, C, D);
    in that order the sum is one batched matrix product that reads the array
    where it lies, where an einsum over the moved axes would copy it first.
    """
    stored = x.permute(0, 1, 4, 2, 3)
    if weights.dim() == 3 and stored.is_contiguous():
        return (weights[:, :, None, None, :] @ stored)[..., 0, :].permute(0, 1, 3, 2)
    if weights.dim() == 3:
        return torch.einsum('bvcdt,bvc->bvdt', x, weights)
    return torch.einsum('bvcdt,bvdc->bvdt', x, weights)


def wsvd_weights(gram, cov, coil_mask, whiten, with_reference):
    """
    Per-coil weights of FSL-MRS wSVD for every sample at once, active coils only.

    Masked coils are cut out of the problem exactly: their rows and columns of
    the covariance are replaced by the identity and those of the reference
    Gram matrix by zeros, so the Cholesky factor, the principal vector and the
    weights all split into an active block - identical to the gathered
    sub-problem - and zeros. The weights do not depend on which whitening
    matrix is used (any W with W W^H = C^-1 gives the same), so the Cholesky
    factor stands in for FSL-MRS's eigendecomposition. The one arbitrary
    choice, the global phase pinned to the first coil, is pinned to the first
    active coil.

    Args:
        gram: X^H X of the reference, (B, ..., C, C).
        cov: Coil noise covariance, (B, C, C).
        coil_mask: Active coils, (B, C) bool.
        whiten: Whether to prewhiten, (B,) bool (FSL-MRS drops it below ten
            noise samples per coil).
        with_reference: True for 'svd_weights' (weights from a reference), False
            for 'svd' (a transient combined with its own decomposition).

    Returns:
        Complex128 weights, (B, ..., C), zero on masked coils; with a single
        active coil, exactly one on it (FSL-MRS leaves such data uncombined).
    """
    return run_steps(wsvd_weight_steps(gram, cov, coil_mask, whiten, with_reference))


def wsvd_weight_steps(gram, cov, coil_mask, whiten, with_reference):
    """"wsvd_weights" as a stepwise computation (see "Step"); the same arithmetic."""
    b, c = coil_mask.shape
    lead = gram.shape[1:-2]
    shape = (b,) + (1,) * len(lead)
    mask = coil_mask.to(torch.float64)
    pair = (mask[:, :, None] * mask[:, None, :]).to(torch.complex128)
    eye = torch.eye(c, dtype=torch.complex128, device=gram.device)
    free = torch.diag_embed((1.0 - mask).to(torch.complex128))

    whitened = cov.to(torch.complex128) * pair + free
    cov_eff = torch.where(whiten[:, None, None], whitened, eye.expand(b, c, c))
    chol = torch.linalg.cholesky_ex(cov_eff).L.reshape(shape + (c, c))

    gram = gram * pair.reshape(shape + (c, c))
    half = torch.linalg.solve_triangular(chol, gram, upper=False)
    whitened_gram = torch.linalg.solve_triangular(chol, half.mH, upper=False).mH
    principal = principal_vector(whitened_gram)                     # (B, ..., C)
    vh0 = principal.conj()

    active = mask.reshape(shape + (c,))
    amp = (vh0[..., None, :] @ chol.mH)[..., 0, :] * active
    first = first_true(coil_mask).reshape(shape + (1,)).expand(*amp.shape[:-1], 1)
    amp0 = amp.gather(-1, first)
    rescale = torch.linalg.vector_norm(amp, dim=-1, keepdim=True) * amp0 / amp0.abs()

    if with_reference:
        # MAGMA's batched solve allocates device memory of its own: no CUDA graph holds it
        solved = yield Step(torch.cholesky_solve, amp.conj()[..., None], chol)
        weights = solved[..., 0] * rescale
    else:
        weights = torch.linalg.solve_triangular(
            chol.mH, principal[..., None], upper=True)[..., 0] * rescale
    weights = weights * active

    single = (coil_mask.sum(dim=-1) == 1).reshape(shape + (1,))
    unit = torch.zeros_like(weights).scatter(-1, first, 1.0)
    return torch.where(single, unit, weights)


def principal_vector(gram, squarings=12, steps=2):
    """
    The eigenvector of the largest eigenvalue of Hermitian PSD matrices, (..., C).

    What FSL-MRS takes from an SVD, without the device synchronisation every
    CUDA eigensolver makes to check its convergence: the matrix is squared
    (and renormalised) *squarings* times, which leaves the dominant direction
    with a weight (l2 / l1)^(2^squarings) on the next - rounding, for any gap a
    reference array shows - and two power steps on the matrix itself restore
    the precision the squarings cost. The phase is arbitrary, as an SVD's is.
    """
    power = gram / torch.linalg.matrix_norm(gram, keepdim=True).clamp(min=1e-300)
    for _ in range(squarings):
        power = power @ power
        power = power / torch.linalg.matrix_norm(power, keepdim=True).clamp(min=1e-300)
    column = torch.linalg.vector_norm(power, dim=-2).argmax(dim=-1)
    vector = power.gather(-1, column[..., None, None].expand(*power.shape[:-1], 1))[..., 0]
    for _ in range(steps):
        vector = (gram @ vector[..., None])[..., 0]
        vector = vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp(min=1e-300)
    return vector


#******************#
#   registration   #
#******************#
#: scipy.optimize.bracket's expansion ratio and growth limit: the first probes of FSL-MRS's Powell.
GOLD = 1.618034
GROW_LIMIT = 110.0


def phasor(angle):
    """exp(i angle) for real *angle*: cos and sin beat a complex exp several times on CPU."""
    if angle.device.type == 'cpu':
        return torch.complex(torch.cos(angle), torch.sin(angle))
    return torch.polar(torch.ones_like(angle), angle)


class ShiftProfile:
    """
    The FSL-MRS alignment cost of every transient, as an exact function of its shift.

    For a transient against the target, both windowed in ppm, the squared cost
    at phase phi and shift nu (cycles per sample) is
    "E(nu) + |Y|^2 - 2 Re(e^{-i phi} K(nu))": K is the in-window
    cross-correlation of the shifted spectrum with the target's, E the
    in-window energy of the shifted spectrum. Both are trigonometric
    polynomials in nu - "K(nu) = sum_n h_n e^{-2 pi i (n + 1) nu}", with h the
    transient (first point halved) times the conjugate time-domain window of
    the target, and "E(nu) = sum_l q_l e^{-2 pi i l nu}", with q its
    autocorrelation times the Dirichlet kernel of the window - so any shift
    costs one pass over the samples and no transform.

    A pass needs "sum_n w_n^j c_n e^{-i theta_n}" for c in (h, q) and moments
    j <= 2. Split into real and imaginary parts that is a set of dot products
    of fixed weighted rows with cos(theta) and sin(theta), so the rows are
    weighted once here and every pass is two batched matrix products - real
    arithmetic, which torch runs several times faster than complex on CPU.

    Args:
        x: Transients, (B, D, T) complex128.
        target: One target per sample, (B, T) complex128.
        first: First bin of the ppm window (FSL-MRS limit_to_range).
        last: Bin after the window's last.
    """

    def __init__(self, x, target, first, last):
        n = x.shape[-1]
        spec = fid_to_spec(target)
        windowed = torch.zeros_like(spec)
        windowed[:, first:last] = spec[:, first:last]
        self.y_energy = (windowed.abs() ** 2).sum(dim=-1)
        self.norm = torch.linalg.vector_norm(target, dim=-1)
        window = math.sqrt(n) * torch.fft.ifft(torch.fft.ifftshift(windowed, dim=-1), dim=-1)
        xt = halve_first(x)
        h = xt * window.conj()[:, None, :]

        # the autocorrelation is exact from the power on a grid of twice the length
        spectrum = torch.fft.fft(xt, n=2 * n, dim=-1)
        autocorr = torch.fft.ifft(spectrum.real ** 2 + spectrum.imag ** 2, dim=-1)[..., :n]
        q = autocorr * constant(('window kernel', n, first, last),
                                lambda: _window_kernel(n, first, last), x.device)
        self.q0 = q[..., 0].real.clone()
        q[..., 0] = 0

        self.lag = torch.arange(n, device=x.device, dtype=torch.float64)
        index = self.lag + 1
        # rows: (h, q) x (re, im) x (moment 0, 1, 2); h is weighted by n + 1, q by n
        rows = []
        for c, w in ((h, index), (q, self.lag)):
            for part in (c.real, c.imag):
                rows += [part, part * w, part * w ** 2]
        self.rows = torch.stack(rows, dim=-2)                       # (B, D, 12, T)
        self.plain = self.rows[..., 0::3, :].contiguous()           # moment 0 only

    def shared(self, nus):
        """K and E at shifts every transient shares, (nus,) -> each (B, D, P)."""
        theta = 2 * math.pi * self.lag[:, None] * nus[None, :]      # (T, P)
        cos = (self.plain @ torch.cos(theta)).movedim(-2, -1)       # (B, D, P, 4)
        sin = (self.plain @ torch.sin(theta)).movedim(-2, -1)
        k, e = self._combine(cos, sin, 1)
        lead = phasor(-2 * math.pi * nus)
        return lead * k[0], self.q0[..., None] + 2 * e[0].real

    def at(self, nu, derivatives=False):
        """
        K and E at one shift per transient, (B, D); with *derivatives*, as
        "((K, K', K''), (E, E', E''))" in nu.
        """
        theta = 2 * math.pi * self.lag * nu[..., None]              # (B, D, T)
        rows = self.rows if derivatives else self.plain
        cos = (rows @ torch.cos(theta)[..., None])[..., 0]
        sin = (rows @ torch.sin(theta)[..., None])[..., 0]
        k, e = self._combine(cos, sin, 3 if derivatives else 1)

        rate = 2 * math.pi
        lead = phasor(-rate * nu)
        k0, e0 = lead * k[0], self.q0 + 2 * e[0].real
        if not derivatives:
            return k0, e0
        return ((k0, lead * k[1] * (-1j * rate), lead * k[2] * (-rate ** 2)),
                (e0, 2 * (e[1] * (-1j * rate)).real, 2 * e[2].real * (-rate ** 2)))

    @staticmethod
    def _combine(cos, sin, moments):
        """
        The complex moment sums from the dot products with cos and sin:
        "(re + i im) e^{-i theta} = (re cos + im sin) + i (im cos - re sin)".
        """
        def sums(offset):
            re = slice(offset, offset + moments)
            im = slice(offset + moments, offset + 2 * moments)
            return torch.complex(cos[..., re] + sin[..., im],
                                 cos[..., im] - sin[..., re]).unbind(dim=-1)
        return sums(0), sums(2 * moments)

    def locked(self, phi, k, e):
        """The squared, unnormalised cost at phase *phi*."""
        return e + _per_sample(self.y_energy, k) - 2 * (phasor(-phi) * k).real

    def cost(self, phi, k, e):
        """FSL-MRS's cost at phase *phi*: the in-window residual norm over the target's."""
        return torch.sqrt(self.locked(phi, k, e).clamp(min=0)) / _per_sample(self.norm, k)


def _window_kernel(n, first, last):
    """w_l = (1/n) sum over the window's bins of e^{-2 pi i l kappa / n}, l = 0 .. n-1."""
    indicator = np.zeros(n)
    indicator[np.fft.fftshift(np.arange(n))[first:last]] = 1.0
    return np.fft.fft(indicator) / n


def _per_sample(value, like):
    """A (B,) value shaped to broadcast over *like*'s (B, ...)."""
    return value.reshape((-1,) + (1,) * (like.dim() - 1))


def _rows_per_chunk(x, bytes_per_point):
    """
    Samples per chunk that keep a work array of *bytes_per_point* per (D, T)
    point below glibc's mmap threshold (32 MB) on CPU, where a fresh large
    allocation pays its page faults on every batch; an accelerator takes the
    whole batch at once.
    """
    if x.device.type != 'cpu':
        return x.shape[0]
    per_sample = bytes_per_point * int(np.prod(x.shape[1:]))
    return max(1, (28 * 2 ** 20) // per_sample)


def _bracket(profile, phi, sw_hz, iterations, base=None):
    """
    scipy.optimize.bracket from (0, 1 Hz) on the phase-locked cost, for all at once.

    A line search of FSL-MRS's Powell along the shift, at the phase that is
    optimal where it starts, opens this bracket, and the bracket decides which
    of the cost's ripples the search settles in. Its loop is emulated with
    masks for a fixed number of *iterations* - real transients need at most
    one - and the points are in Hz relative to where the search starts, as
    there.

    Args:
        profile: The ShiftProfile.
        phi: The locked phase, (B, D).
        sw_hz: Spectral width in Hz.
        iterations: Expansion steps emulated.
        base: Where the search starts, (B, D) in cycles per sample; None is
            zero, whose probes every transient shares.

    Returns:
        "(xa, xb, xc, fa, fb, fc, open)": the bracket in Hz from *base*, its
        costs, and where it was still expanding when the iterations ran out.
    """
    def cost(eps):
        nu = eps / sw_hz if base is None else base + eps / sw_hz
        return profile.cost(phi, *profile.at(nu))

    if base is None:
        probes = constant(('bracket probes', sw_hz),
                          lambda: np.array([0.0, 1.0, 1.0 + GOLD, -GOLD]) / sw_hz, phi.device)
        k, e = profile.shared(probes)
        f0, f1, f_pos, f_neg = profile.cost(phi[..., None], k, e).unbind(dim=-1)
        swap = f0 < f1
    else:
        f0, f1 = cost(torch.zeros_like(base)), cost(torch.ones_like(base))
        swap = f0 < f1
    xa = swap.to(f0.dtype)
    xb = 1.0 - xa
    xc = (1.0 + GOLD) - (1.0 + 2 * GOLD) * xa
    fa = torch.where(swap, f1, f0)
    fb = torch.where(swap, f0, f1)
    fc = torch.where(swap, f_neg, f_pos) if base is None else cost(xc)
    expanding = fc < fb

    for _ in range(iterations):
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        denom = torch.where(val.abs() < 1e-21, 2e-21, 2.0 * val)
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + GROW_LIMIT * (xc - xb)
        inside = (w - xc) * (xb - w) > 0
        limit = ~inside & ((w - wlim) * (wlim - xc) >= 0)
        beyond = ~inside & ~limit & ((w - wlim) * (xc - w) > 0)
        p1 = torch.where(inside | beyond, w,
                         torch.where(limit, wlim, xc + GOLD * (xc - xb)))
        f_p1 = cost(p1)

        # a minimum between b and c, or a point above b, closes the bracket
        take_b = inside & (f_p1 < fc)
        take_c = inside & ~take_b & (f_p1 > fb)
        closed = take_b | take_c
        # past c and still falling: step once more before extending
        falling = beyond & (f_p1 < fc)
        second = (inside & ~closed) | falling
        mid_b, mid_c = torch.where(falling, xc, xb), torch.where(falling, p1, xc)
        mid_fb, mid_fc = torch.where(falling, fc, fb), torch.where(falling, f_p1, fc)
        p2 = mid_c + GOLD * (mid_c - mid_b)
        f_p2 = cost(p2)
        w = torch.where(second, p2, p1)
        fw = torch.where(second, f_p2, f_p1)

        na = torch.where(closed, torch.where(take_b, xb, xa), mid_b)
        nb = torch.where(closed, torch.where(take_b, p1, xb), mid_c)
        nc = torch.where(closed, torch.where(take_c, p1, xc), w)
        nfa = torch.where(closed, torch.where(take_b, fb, fa), mid_fb)
        nfb = torch.where(closed, torch.where(take_b, f_p1, fb), mid_fc)
        nfc = torch.where(closed, torch.where(take_c, f_p1, fc), fw)
        xa, xb, xc = (torch.where(expanding, new, old)
                      for new, old in ((na, xa), (nb, xb), (nc, xc)))
        fa, fb, fc = (torch.where(expanding, new, old)
                      for new, old in ((nfa, fa), (nfb, fb), (nfc, fc)))
        expanding = expanding & ~closed & (fc < fb)

    return xa, xb, xc, fa, fb, fc, expanding


#: scipy's Brent: golden-section fraction and minimal tolerance; Powell line searches use tol 1e-2.
CGOLD = 0.3819660
BRENT_MINTOL = 1.0e-11
BRENT_TOL = 1.0e-2


def _brent(profile, phi, sw_hz, bracket, base, iterations):
    """
    scipy's Brent minimisation of the phase-locked cost inside *bracket*, all at once.

    Powell's line search hands the bracket to Brent, whose first golden-section
    steps decide which minimum inside it the search ends in; the iterations are
    emulated exactly (in Hz from *base*, as there) with masks, a transient that
    converged staying put.

    Args:
        profile: The ShiftProfile.
        phi: The locked phase, (B, D).
        sw_hz: Spectral width in Hz.
        bracket: "(xa, xb, xc, fb)" from "_bracket".
        base: Where the line search starts, (B, D) in cycles per sample, or None.
        iterations: Brent iterations emulated.

    Returns:
        "(x, a, b)": the minimum found and the interval left around it, in Hz from *base*.
    """
    xa, xb, xc, fb = bracket

    def cost(eps):
        nu = eps / sw_hz if base is None else base + eps / sw_hz
        return profile.cost(phi, *profile.at(nu))

    x = w = v = xb
    fx = fw = fv = fb
    a, b = torch.minimum(xa, xc), torch.maximum(xa, xc)
    deltax = torch.zeros_like(xb)
    rat = torch.zeros_like(xb)
    done = torch.zeros_like(xb, dtype=torch.bool)
    for _ in range(iterations):
        tol1 = BRENT_TOL * x.abs() + BRENT_MINTOL
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        done = done | ((x - xmid).abs() < (tol2 - 0.5 * (b - a)))
        far = torch.where(x >= xmid, a - x, b - x)

        # a parabola through x, w, v, where it is usable; a golden section otherwise
        tmp1 = (x - w) * (fx - fv)
        tmp2 = (x - v) * (fx - fw)
        p = (x - v) * tmp2 - (x - w) * tmp1
        q = 2.0 * (tmp2 - tmp1)
        p = torch.where(q > 0, -p, p)
        q = q.abs()
        usable = ((p > q * (a - x)) & (p < q * (b - x))
                  & (p.abs() < (0.5 * q * deltax).abs()))
        golden_first = deltax.abs() <= tol1
        parabolic = ~golden_first & usable
        step = torch.where(parabolic, p / torch.where(q > 0, q, 1.0), CGOLD * far)
        near_edge = ((x + step - a) < tol2) | ((b - x - step) < tol2)
        step = torch.where(parabolic & near_edge,
                           torch.where(xmid - x >= 0, tol1, -tol1), step)
        deltax = torch.where(golden_first | ~usable, far, rat)
        rat = step
        u = x + torch.where(rat.abs() < tol1, torch.where(rat >= 0, tol1, -tol1), rat)
        fu = cost(u)

        worse = fu > fx
        a_new = torch.where(worse, torch.where(u < x, u, a), torch.where(u >= x, x, a))
        b_new = torch.where(worse, torch.where(u < x, b, u), torch.where(u >= x, b, x))
        shift_w = worse & ((fu <= fw) | (w == x))
        shift_v = worse & ~shift_w & ((fu <= fv) | (v == x) | (v == w))
        v_new = torch.where(worse, torch.where(shift_w, w, torch.where(shift_v, u, v)), w)
        fv_new = torch.where(worse, torch.where(shift_w, fw, torch.where(shift_v, fu, fv)), fw)
        w_new = torch.where(worse, torch.where(shift_w, u, w), x)
        fw_new = torch.where(worse, torch.where(shift_w, fu, fw), fx)
        x_new = torch.where(worse, x, u)
        fx_new = torch.where(worse, fx, fu)

        keep = done
        a, b = torch.where(keep, a, a_new), torch.where(keep, b, b_new)
        v, fv = torch.where(keep, v, v_new), torch.where(keep, fv, fv_new)
        w, fw = torch.where(keep, w, w_new), torch.where(keep, fw, fw_new)
        x, fx = torch.where(keep, x, x_new), torch.where(keep, fx, fx_new)
    return x, a, b


def _newton(profile, nu, low, high, step_cap, steps, phi=None):
    """
    Safeguarded Newton descent on the cost inside [low, high] (cycles per sample).

    With *phi* the phase stays locked there; without, it is optimal at every
    shift, which leaves "E + |Y|^2 - 2 |K|" to minimise. Where the curvature is
    not positive the step goes downhill by the cap instead.
    """
    for _ in range(steps):
        (k, k1, k2), (_, e1, e2) = profile.at(nu, derivatives=True)
        if phi is not None:
            turn = phasor(-phi)
            grad = e1 - 2 * (turn * k1).real
            hess = e2 - 2 * (turn * k2).real
        else:
            mag = k.abs().clamp(min=1e-300)
            slope = (k.conj() * k1).real / mag
            curve = ((k1.abs() ** 2 + (k.conj() * k2).real) - slope ** 2) / mag
            grad = e1 - 2 * slope
            hess = e2 - 2 * curve
        step = torch.where(hess > 0, -grad / hess, -torch.sign(grad) * step_cap)
        nu = torch.minimum(torch.maximum(nu + step.clamp(-step_cap, step_cap), low), high)
    return nu


def align(fids, mask, sw_hz, sf_mhz, ppmlim, **options):
    """
    Phase and frequency shifts aligning transients, on the FSL-MRS objective.

    FSL-MRS phase_freq_align minimises
    "|| extract(e^{-i phi} shift(FID, eps)) - extract(target) || / || target ||"
    per transient with a Powell search from zero, against the transient
    nearest the mean. On noisy transients that cost ripples on the scale of a
    spectral bin, so which minimum is found depends on the search path; the
    path is therefore followed where it decides, and solved exactly where it
    does not:

    1. Powell's first line search, along the phase at zero shift, ends at the
       closed-form optimal phase.
    2. Its second, along the shift at that phase, opens scipy's bracket from
       (0, 1 Hz) and hands it to Brent, whose first golden-section steps pick
       the minimum inside it; both are emulated exactly, and an optional
       Newton descent on the locked cost continues from there.
    3. The remaining Powell iterations free the phase and converge to the
       nearest stationary point of the full cost; a Newton descent on the
       phase-optimal profile does the same within a bin.

    FSL-MRS runs Powell twice, the second time from the first result, and the
    second bracket can open into a deeper ripple next door; so the same three
    steps run once more from there, and the phase is read off in closed form.
    Every cost is evaluated exactly (see "ShiftProfile").

    Args:
        fids: Transients, (B, D, T) complex.
        mask: Transients to align, (B, D) bool; the others are left at zero
            shift. A sample with fewer than two is not aligned, as in FSL-MRS.
        sw_hz: Spectral width in Hz.
        sf_mhz: Spectrometer frequency in MHz.
        ppmlim: ppm window of the comparison.
        passes: Powell runs emulated (FSL-MRS's niter).
        bracket_iterations: Expansion steps of each emulated bracket.
        brent_iterations: Brent steps emulated inside each bracket.
        locked_steps: Newton steps on the locked cost after them.
        free_steps: Newton steps with the phase free, within a bin, per pass
            (the last entry for any further pass).
        max_shift_hz: Bound on the shift (default a quarter of the spectral
            width).
        spans: The (first, last) bins of *ppmlim*, where the caller has them;
            *sf_mhz* and *ppmlim* are then not read.

    Returns:
        "(phi, eps)" in radians and Hz, (B, D) float64.
    """
    rows = _rows_per_chunk(fids, 12 * 8)
    if rows >= fids.shape[0]:
        return _align(fids, mask, sw_hz, sf_mhz, ppmlim, **options)
    parts = [_align(f, m, sw_hz, sf_mhz, ppmlim, **options)
             for f, m in zip(fids.split(rows), mask.split(rows))]
    return tuple(torch.cat(values) for values in zip(*parts))


def _align(fids, mask, sw_hz, sf_mhz, ppmlim, passes=2, bracket_iterations=1,
           brent_iterations=2, locked_steps=0, free_steps=(2, 3), max_shift_hz=None,
           spans=None):
    """"align" on one chunk of samples."""
    x = fids.to(torch.complex128)
    b, d, n = x.shape
    first, last = spans if spans is not None else ppm_window(n, sw_hz, sf_mhz, ppmlim)
    reach = (sw_hz / 4 if max_shift_hz is None else max_shift_hz) / sw_hz

    # the target: the transient nearest the mean of the valid ones, first of any tie
    weights = mask.to(torch.float64)
    avg = (x * weights[..., None]).sum(dim=1, keepdim=True) / weights.sum(dim=1)[:, None, None]
    dist = torch.linalg.vector_norm(x - avg, dim=-1).masked_fill(~mask, torch.inf)
    near = dist <= dist.min(dim=-1, keepdim=True).values * (1 + 1e-9)
    target = x.gather(1, first_true(near)[:, None, None].expand(b, 1, n))[:, 0]

    profile = ShiftProfile(x, target, first, last)
    nu = torch.zeros(b, d, dtype=torch.float64, device=x.device)
    k, _ = profile.at(nu)
    for step in range(passes):
        # the phase line search ends at the phase that is optimal where the pass starts
        phi = torch.angle(k)
        base = None if step == 0 else nu
        xa, xb, xc, fa, fb, fc, expanding = _bracket(profile, phi, sw_hz, bracket_iterations,
                                                     base)
        valid = ((((fb < fc) & (fb <= fa)) | ((fb < fa) & (fb <= fc)))
                 & (((xa < xb) & (xb < xc)) | ((xc < xb) & (xb < xa))))
        best = torch.stack([xa, xb, xc]).gather(
            0, torch.stack([fa, fb, fc]).argmin(dim=0)[None])[0]
        # an invalid bracket leaves scipy at its best point; one cut short still
        # falls towards c, so that side stays open
        offset = nu if step else 0.0
        usable = valid | expanding
        start = offset + torch.where(usable, xb, best) / sw_hz
        low = torch.where(usable, offset + torch.minimum(xa, xc) / sw_hz, start)
        high = torch.where(usable, offset + torch.maximum(xa, xc) / sw_hz, start)
        low = torch.where(expanding & (xc < xa), -reach, low)
        high = torch.where(expanding & (xc > xa), reach, high)

        if brent_iterations:
            found, left, right = _brent(profile, phi, sw_hz, (xa, xb, xc, fb), base,
                                        brent_iterations)
            start = torch.where(valid, offset + found / sw_hz, start)
            low = torch.where(valid, offset + left / sw_hz, low)
            high = torch.where(valid, offset + right / sw_hz, high)
        nu = _newton(profile, start, low, high, 0.5 / sw_hz, locked_steps, phi=phi)
        steps = free_steps[min(step, len(free_steps) - 1)]
        nu = _newton(profile, nu, nu - 1.0 / n, nu + 1.0 / n, 0.5 / sw_hz, steps)
        k, _ = profile.at(nu)

    phi = torch.angle(k)
    eps = nu * sw_hz
    moving = mask & (mask.sum(dim=-1, keepdim=True) > 1)
    return torch.where(moving, phi, 0.0), torch.where(moving, eps, 0.0)


def alignment_phasor(phi, eps, n, sw_hz, dtype):
    """e^{-i phi} e^{-2 pi i t eps} on FSL-MRS's time axis (dwell .. n dwell), (..., T)."""
    t = torch.linspace(1.0 / sw_hz, n / sw_hz, n, dtype=torch.float64, device=phi.device)
    angle = -phi[..., None] - 2 * math.pi * t * eps[..., None]
    return torch.polar(torch.ones_like(angle), angle).to(dtype)


#***********************#
#   outlier detection   #
#***********************#
def unlike_mask(fids, mask, sdlimit=1.96, niter=2):
    """
    Keep-mask of FSL-MRS identifyUnlikeFIDs (ppmlim=None) over valid transients.

    The metric is the distance of every spectrum to the spectrum of the
    complex median; the spectra are a unitary transform of the FIDs with their
    first point halved, so the distance is taken in the time domain instead
    (Parseval) and no transform is needed. Mean and standard deviation run
    over the valid transients, as FSL-MRS's run over the ones it was given.
    Everything is real arithmetic, and the medians sort the data in its own
    precision - the middle values are exact either way.

    Args:
        fids: Transients, (B, D, T) complex.
        mask: Valid transients, (B, D) bool, at least one per sample.
        sdlimit: Exclusion limit in standard deviations.
        niter: Target-refinement iterations.

    Returns:
        Boolean keep mask, (B, D): valid and alike.
    """
    b, d, n = fids.shape
    parts = torch.view_as_real(fids).permute(0, 2, 3, 1).contiguous()    # (B, T, 2, D)
    halved = torch.view_as_real(halve_first(fids.to(torch.complex128))).reshape(b, d, 2 * n)
    energy = (halved ** 2).sum(dim=-1)
    weights = mask.to(torch.float64)
    count = weights.sum(dim=-1, keepdim=True)

    keep = mask
    target = _real_median(parts, mask)
    for step in range(niter):
        target[:, 0] *= 0.5                                         # its spectrum's first point
        flat = target.reshape(b, 2 * n)
        cross = (halved @ flat[..., None])[..., 0]
        distance = energy - 2 * cross + (flat ** 2).sum(dim=-1, keepdim=True)
        metric = torch.sqrt(distance.clamp(min=0))
        avg = (metric * weights).sum(dim=-1, keepdim=True) / count
        std = torch.sqrt((((metric - avg) ** 2) * weights).sum(dim=-1, keepdim=True) / count)
        keep = mask & ((metric - avg).abs() <= sdlimit * std)
        if step < niter - 1:
            target = _real_median(parts, keep)
    return keep


def _real_median(parts, mask):
    """
    NumPy's median over the valid transients of (B, T, 2, D) parts, in float64 (B, T, 2).

    The mean of the two middle values for an even count - torch.median would
    return the lower one - with the invalid entries sorted behind the valid.
    """
    count = mask.sum(dim=-1)
    ordered = torch.where(mask[:, None, None, :], parts, torch.inf).sort(dim=-1).values
    shape = ordered.shape[:-1] + (1,)
    low = ((count - 1) // 2).reshape(-1, 1, 1, 1).expand(shape)
    high = (count // 2).reshape(-1, 1, 1, 1).expand(shape)
    pair = ordered.gather(-1, low).to(torch.float64) + ordered.gather(-1, high).to(torch.float64)
    return 0.5 * pair[..., 0]


#***************#
#   ecc phase   #
#***************#
def unwrap(phase):
    """numpy.unwrap along the last axis (discontinuity pi, period 2 pi)."""
    diff = phase[..., 1:] - phase[..., :-1]
    wrapped = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
    wrapped = torch.where((wrapped == -math.pi) & (diff > 0), math.pi, wrapped)
    correct = torch.where(diff.abs() < math.pi, 0.0, wrapped - diff)
    return torch.cat([phase[..., :1], phase[..., 1:] + torch.cumsum(correct, dim=-1)], dim=-1)


def ecc_phase(refs, width=32):
    """
    Smoothed unwrapped phase of the reference FIDs, (..., T).

    suspect's sliding_gaussian as nifti_ecc_smoothed uses it: edge-padded with
    10-point edge means, correlated with a normalised Gaussian window.
    """
    phase = unwrap(torch.angle(refs.to(torch.complex128)))
    lead, n = phase.shape[:-1], phase.shape[-1]
    flat = phase.reshape(-1, n)
    window = torch.exp(-torch.linspace(-3, 3, width, dtype=torch.float64,
                                       device=phase.device) ** 2)
    window = window / window.sum()
    offset = (width - 1) // 2
    left = flat[:, :10].mean(dim=-1, keepdim=True).expand(-1, offset)
    right = flat[:, -10:].mean(dim=-1, keepdim=True).expand(-1, width - 1 - offset)
    padded = torch.cat([left, flat, right], dim=-1)
    smooth = torch.nn.functional.conv1d(padded[:, None, :], window[None, None, :])[:, 0]
    return smooth.reshape(lead + (n,))


#*******************#
#   peak searches   #
#*******************#
def _padded_window(fids, sw_hz, sf_mhz, window, spans=None):
    """
    The four-fold zero-filled spectrum inside *window* (ppm), and that window's
    bounds; *spans* gives the bounds directly (then *sf_mhz* is not read).
    """
    n = fids.shape[-1]
    padded = torch.cat([fids, torch.zeros(fids.shape[:-1] + (3 * n,), dtype=fids.dtype,
                                          device=fids.device)], dim=-1)
    first, last = spans if spans is not None else ppm_window(4 * n, sw_hz, sf_mhz, window)
    return fid_to_spec(padded.to(torch.complex128))[..., first:last], first, last


def peak_phase(fids, sw_hz, sf_mhz, window, spans=None):
    """
    Zero-order phase of FSL-MRS phaseCorrect: minus the angle at the window's
    peak, (...); *spans* as for "_padded_window".
    """
    spec, _, _ = _padded_window(fids, sw_hz, sf_mhz, window, spans)
    peak = spec.abs().argmax(dim=-1, keepdim=True)
    return -torch.angle(spec.gather(-1, peak))[..., 0]


def peak_shift_hz(fids, sw_hz, sf_mhz, window, reference_ppm):
    """The shift of FSL-MRS shiftToRef: window peak minus the reference, in Hz, (...)."""
    spans = ppm_window(4 * fids.shape[-1], sw_hz, sf_mhz, window)
    return run_steps(peak_shift_steps(fids, sw_hz, spans, reference_ppm), {'sf_mhz': sf_mhz})


def peak_shift_steps(fids, sw_hz, spans, reference_ppm):
    """
    "peak_shift_hz" on the window's bounds, as a stepwise computation.

    The ppm axis and the spectrometer frequency the peak is read against
    differ from scan to scan, so that last conversion is a step, late in
    'sf_mhz'.
    """
    spec, first, last = _padded_window(fids, sw_hz, None, None, spans)
    peak = spec.abs().argmax(dim=-1)
    return (yield Step(_peak_hz, peak, n=4 * fids.shape[-1], sw_hz=sw_hz, spans=(first, last),
                       reference_ppm=reference_ppm, late=('sf_mhz',)))


def _peak_hz(peak, n, sw_hz, spans, reference_ppm, sf_mhz):
    """The ppm of the peak bins *peak* on the shifted axis, less the reference, in Hz."""
    first, last = spans
    axis = constant(('ppm axis', n, sw_hz, sf_mhz, first, last),
                    lambda: ppm_shift_axis(n, sw_hz, sf_mhz)[first:last], peak.device)
    return (axis[peak] - reference_ppm) * sf_mhz


def shift_phasor(shift_hz, n, sw_hz, dtype):
    """e^{-2 pi i t shift} on FSL-MRS freqshift's time axis (0 .. n dwell), (..., T)."""
    t = torch.linspace(0, n / sw_hz, n, dtype=torch.float64, device=shift_hz.device)
    angle = -2 * math.pi * t * shift_hz[..., None]
    return torch.polar(torch.ones_like(angle), angle).to(dtype)
