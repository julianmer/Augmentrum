####################################################################################################
#                                       dataset_utils.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Backend-agnostic dataset utilities for Augmentrum.                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import random
import numpy as np
from typing import List, Optional, Callable

# internal
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.sampling.subject_splitter import SubjectSplitter


__all__ = [
    'create_random_generator',
    'create_fixed_generator',
    'convert_batch_to_backend',
    'wrap_generator_for_framework',
    'output_tokens',
    'build_outputs',
    'map_structure',
    'flatten_structure',
    'SubjectSplitter',
]


#**********************#
#   outputs handling   #
#**********************#
def output_tokens(spec) -> List[str]:
    """All string tokens in a nested outputs spec, left to right."""
    if isinstance(spec, str):
        return [spec]
    tokens = []
    for element in spec:
        tokens.extend(output_tokens(element))
    return tokens


def build_outputs(spec, data, water, taps):
    """
    Fill a nested outputs spec with the pipeline stages it names.

    Tokens: 'data' / 'water' are the pipeline end; '<tap>' / '<tap>.water' are
    a tapped stage. The returned structure mirrors the spec, tuples throughout.
    """
    if isinstance(spec, str):
        if spec == 'data':
            return data
        if spec == 'water':
            return water
        name, _, member = spec.partition('.')
        if name in taps and member in ('', 'water'):
            return taps[name][0] if member == '' else taps[name][1]
        raise ValueError(
            f"outputs token '{spec}' names no pipeline stage. Available: 'data', "
            f"'water', and per tap '<name>' / '<name>.water' for taps {list(taps)}."
        )
    return tuple(build_outputs(element, data, water, taps) for element in spec)


def map_structure(fn, structure):
    """Apply *fn* to every leaf of a nested tuple/list structure."""
    if isinstance(structure, (tuple, list)):
        return tuple(map_structure(fn, element) for element in structure)
    return fn(structure)


def flatten_structure(structure) -> List:
    """The leaves of a nested tuple/list structure, left to right."""
    if not isinstance(structure, (tuple, list)):
        return [structure]
    flat = []
    for element in structure:
        flat.extend(flatten_structure(element))
    return flat


def _resolve_outputs(pipeline, result, outputs):
    """
    Shape one pipeline result for yielding.

    A pipeline with taps returns "(data, water, taps)"; without, "(data, water)".
    No spec keeps the classic "(data, water)" pair either way — taps only reach
    the loader through an explicit outputs spec.
    """
    if pipeline.has_taps:
        data, water, taps = result
    else:
        data, water = result
        taps = {}
    if outputs is None:
        return data, water
    return build_outputs(outputs, data, water, taps)


#*********************************#
#   backend-agnostic generators   #
#*********************************#
def _make_batch(data, water, indices, copy=False):
    """
    Build a (batch_data, batch_water) NIfTI_MRS_Plus pair from subject indices.

    Args:
        data:    NIfTI_MRS_Plus pool of subjects.
        water:   Optional NIfTI_MRS_Plus pool of water references.
        indices: List of integer subject indices for this batch.
        copy:    If True, copy each NIfTI-MRS object (use for fixed mode
                 where the same objects are yielded multiple times).

    Returns:
        (batch_data, batch_water) as NIfTI_MRS_Plus objects.
    """
    from nifti_mrs_plus import NIfTI_MRS_Plus

    get = (lambda idx: data[idx].copy()) if copy else (lambda idx: data[idx])
    batch_data = NIfTI_MRS_Plus(
        [get(i) for i in indices], backend=data.backend, volatile=data.volatile,
    )

    batch_water = None
    if water is not None:
        get_w = (lambda idx: water[idx].copy()) if copy else (lambda idx: water[idx])
        batch_water = NIfTI_MRS_Plus(
            [get_w(i) for i in indices], backend=water.backend, volatile=water.volatile,
        )

    return batch_data, batch_water


def create_random_generator(data: NIfTI_MRS_Plus,
                            water: Optional[NIfTI_MRS_Plus],
                            pipeline,
                            batch_size: int,
                            outputs=None):
    """
    Infinite random-sampling generator (on-the-fly mode).

    Each iteration: pick *batch_size* random subjects, sample fresh
    augmentation parameters, apply the pipeline, yield the batch.

    Yields:
        (batch_data, batch_water) — NIfTI_MRS_Plus objects — or, with an
        *outputs* spec, the spec's nested structure filled with the stages it
        names.
    """
    n_subjects = len(data)

    while True:
        indices = [random.randint(0, n_subjects - 1) for _ in range(batch_size)]
        batch_data, batch_water = _make_batch(data, water, indices)
        batch_params = pipeline.sample_batch_parameters(batch_size)
        result = pipeline(batch_data, batch_water, batch_params=batch_params)
        yield _resolve_outputs(pipeline, result, outputs)


def create_fixed_generator(data: NIfTI_MRS_Plus,
                           water: Optional[NIfTI_MRS_Plus],
                           pipeline,
                           batch_size: int,
                           shuffle: bool = False,
                           outputs=None):
    """
    Single-pass generator with fixed augmentation parameters.

    Parameters are sampled **once** at creation and reused for every batch,
    giving consistent results across epochs.

    Yields:
        (batch_data, batch_water) — NIfTI_MRS_Plus objects — or, with an
        *outputs* spec, the spec's nested structure filled with the stages it
        names.
    """
    fixed_params = pipeline.sample_batch_parameters(batch_size)

    n_subjects = len(data)
    indices = list(range(n_subjects))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, n_subjects, batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_data, batch_water = _make_batch(data, water, batch_idx, copy=True)
        params = _trim_batch_params(fixed_params, len(batch_idx))
        result = pipeline(batch_data, batch_water, batch_params=params)
        yield _resolve_outputs(pipeline, result, outputs)


def _trim_batch_params(batch_params, n):
    """
    Per-sample vectors cut to a short final batch; scalars pass untouched.

    Parameters are sampled once for the full batch size, but the last batch of
    a single pass carries the remainder of the subjects, and a vector longer
    than the batch would not broadcast.
    """
    return {
        step: {name: value[:n] if isinstance(value, np.ndarray) and value.ndim else value
               for name, value in params.items()}
        for step, params in batch_params.items()
    }


#**********************************#
#   backend conversion functions   #
#**********************************#
def convert_batch_to_backend(batch_data: List, batch_water: List, backend: Backend):
    """
    Convert batch of NIFTI_MRS objects to target backend format.

    Args:
        batch_data: List of NIFTI_MRS or data arrays
        batch_water: List of NIFTI_MRS or None
        backend: Target backend

    Returns:
        (converted_data, converted_water)
    """
    if backend == Backend.NIFTI_LIST:
        return batch_data, batch_water

    elif backend == Backend.NUMPY:
        # Convert to numpy arrays
        data_arrays = []
        for item in batch_data:
            if hasattr(item, '__array__'):
                data_arrays.append(np.array(item))
            elif isinstance(item, NIfTI_MRS_Plus):
                data_arrays.append(item.numpy())
            else:
                # Assume it's a NIFTI_MRS
                data_arrays.append(item[:])

        water_arrays = None
        if batch_water and batch_water[0] is not None:
            water_arrays = []
            for item in batch_water:
                if hasattr(item, '__array__'):
                    water_arrays.append(np.array(item))
                elif isinstance(item, NIfTI_MRS_Plus):
                    water_arrays.append(item.numpy())
                else:
                    water_arrays.append(item[:])
            water_arrays = np.array(water_arrays) if water_arrays else None

        return np.array(data_arrays), water_arrays

    elif backend == Backend.PYTORCH:
        import torch
        # Convert to torch tensors
        data_np, water_np = convert_batch_to_backend(batch_data, batch_water, Backend.NUMPY)
        data_torch = torch.from_numpy(data_np) if data_np is not None else None
        water_torch = torch.from_numpy(water_np) if water_np is not None else None
        return data_torch, water_torch

    elif backend in [Backend.TENSORFLOW, Backend.KERAS]:
        import tensorflow as tf
        # Convert to tensorflow tensors
        data_np, water_np = convert_batch_to_backend(batch_data, batch_water, Backend.NUMPY)
        data_tf = tf.convert_to_tensor(data_np) if data_np is not None else None
        water_tf = tf.convert_to_tensor(water_np) if water_np is not None else None
        return data_tf, water_tf

    elif backend == Backend.JAX:
        import jax.numpy as jnp
        # Convert to jax arrays
        data_np, water_np = convert_batch_to_backend(batch_data, batch_water, Backend.NUMPY)
        data_jax = jnp.array(data_np) if data_np is not None else None
        water_jax = jnp.array(water_np) if water_np is not None else None
        return data_jax, water_jax

    else:
        raise ValueError(f"Unknown backend: {backend}")


#***********************************#
#   framework dataloader wrappers   #
#***********************************#
def wrap_generator_for_framework(generator: Callable,
                                 backend: Backend,
                                 framework: str = None,
                                 structured: bool = False):
    """
    Wrap a generator for a specific framework (PyTorch DataLoader, tf.data, etc.).

    Args:
        generator: Generator (iterator) yielding (data, water) batches — or,
                   with *structured*, arbitrary nested tuples of stages.
        backend: Target backend for data conversion
        framework: 'pytorch', 'tensorflow', 'keras', 'jax', 'numpy', None (raw)
        structured: The generator yields an outputs-spec structure rather than
                    the classic (data, water) pair; every leaf (NIfTI_MRS_Plus
                    or None) is converted in place of the pairwise handling.

    Returns:
        Framework-specific dataloader/dataset/generator
    """
    # Default framework from backend
    if framework is None:
        backend_name = backend.name.lower()
        # nifti_list backend should default to raw python iteration
        if backend_name == 'nifti_list':
            framework = 'python'
        else:
            framework = backend_name

    if structured:
        return _wrap_structured_generator(generator, framework)

    return _wrap_pair_generator(generator, backend, framework)


def _wrap_structured_generator(generator, framework: str):
    """
    Convert every leaf of an outputs-spec structure for the framework.

    The structure mirrors the user's spec exactly; leaves are NIfTI_MRS_Plus
    objects (or None for an absent water reference) and are converted with the
    same per-backend logic as the classic pair path.
    """
    if framework in ['python', 'nifti_list']:
        # Copy each leaf before unwrapping: output and taps share one nifti
        # list on the volatile fast path, and materializing one stage would
        # overwrite the objects just handed out for another. The copy captures
        # each stage's values at its own materialization.
        def structured_gen():
            for batch in generator:
                yield map_structure(
                    lambda leaf: leaf.copy().list() if leaf is not None else None,
                    batch)
        return structured_gen()

    targets = {'numpy': Backend.NUMPY, 'jax': Backend.JAX, 'pytorch': Backend.PYTORCH}
    if framework in targets:
        target = targets[framework]

        def convert_leaf(leaf):
            if leaf is None:
                return None
            converted, _ = convert_batch_to_backend(leaf, None, target)
            return converted

        def structured_gen():
            for batch in generator:
                yield map_structure(convert_leaf, batch)
        return structured_gen()

    raise NotImplementedError(
        f"framework '{framework}' does not support custom outputs structures yet. "
        f"Use framework='numpy' and convert, or the classic (data, water) outputs."
    )


def _wrap_pair_generator(generator, backend: Backend, framework: str):
    """The classic path: batches are (data, water) pairs."""
    # For nifti_list or raw python: unwrap NIfTI_MRS_Plus to raw NIFTI_MRS lists
    if framework in ['python', 'nifti_list']:
        def unwrapped_gen():
            for batch_data, batch_water in generator:
                # Unwrap NIfTI_MRS_Plus objects to raw NIFTI_MRS lists
                if isinstance(batch_data, list):
                    # Batch is a list of NIfTI_MRS_Plus objects, unwrap each
                    data_list = []
                    for item in batch_data:
                        if hasattr(item, 'list'):
                            # It's a NIfTI_MRS_Plus, get the internal list
                            data_list.extend(item.list())
                        else:
                            # Already a NIFTI_MRS object
                            data_list.append(item)
                    batch_data = data_list
                elif hasattr(batch_data, 'list'):
                    # Single NIfTI_MRS_Plus object
                    batch_data = batch_data.list()

                if batch_water is not None:
                    if isinstance(batch_water, list):
                        water_list = []
                        for item in batch_water:
                            if hasattr(item, 'list'):
                                water_list.extend(item.list())
                            else:
                                water_list.append(item)
                        batch_water = water_list
                    elif hasattr(batch_water, 'list'):
                        batch_water = batch_water.list()

                yield batch_data, batch_water
        return unwrapped_gen()

    # For numpy/jax: return generator directly with backend conversion
    if framework in ['numpy', 'jax']:
        def converted_gen():
            for batch_data, batch_water in generator:  # generator is already iterator
                yield convert_batch_to_backend(batch_data, batch_water, backend)
        return converted_gen()

    # For PyTorch: wrap in DataLoader-like iterator
    elif framework == 'pytorch':
        def converted_gen():
            for batch_data, batch_water in generator:  # generator is already iterator
                yield convert_batch_to_backend(batch_data, batch_water, Backend.PYTORCH)
        return converted_gen()

    # For TensorFlow/Keras: wrap in tf.data.Dataset
    elif framework in ['tensorflow', 'keras']:
        try:
            import tensorflow as tf

            def tf_gen():
                for batch_data, batch_water in generator:  # generator is already iterator
                    data_tf, water_tf = convert_batch_to_backend(
                        batch_data, batch_water, Backend.TENSORFLOW
                    )
                    yield data_tf, water_tf

            # Infer output signature from first batch
            # We need to peek at the generator - convert to list first
            batch_list = list(generator)
            if not batch_list:
                raise ValueError("Empty generator")

            first_data, first_water = batch_list[0]
            data_shape = (None,) + first_data[0][:].shape if hasattr(first_data[0], 'shape') else (None, None)

            output_signature = (
                tf.TensorSpec(shape=data_shape, dtype=tf.complex64),
                tf.TensorSpec(shape=data_shape, dtype=tf.complex64) if first_water else None
            )

            # Create generator from list
            def list_gen():
                for batch_data, batch_water in batch_list:
                    data_tf, water_tf = convert_batch_to_backend(
                        batch_data, batch_water, Backend.TENSORFLOW
                    )
                    yield data_tf, water_tf

            return tf.data.Dataset.from_generator(
                list_gen,
                output_signature=output_signature
            )
        except ImportError:
            print("Warning: TensorFlow not installed, returning numpy generator")
            return wrap_generator_for_framework(generator, backend, 'numpy')

    else:
        raise ValueError(f"Unknown framework: {framework}")
