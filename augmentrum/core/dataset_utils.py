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
from typing import List, Optional, Tuple, Dict, Callable, Union

# internal
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.sampling.subject_splitter import SubjectSplitter


__all__ = [
    'create_random_generator',
    'create_fixed_generator',
    'create_deterministic_generator',
    'create_deterministic_indices',
    'convert_batch_to_backend',
    'wrap_generator_for_framework',
    'SubjectSplitter',
]


#**********************#
#   helper functions   #
#**********************#

def create_deterministic_indices(data: NIfTI_MRS_Plus, n_coils: Tuple, n_averages: Tuple) -> List[Tuple]:
    """
    Create all possible (subject_idx, coil_idx, average_idx) combinations.

    Args:
        data: NIfTI_MRS_Plus containing subjects
        n_coils: (min, max) coils to sample
        n_averages: (min, max) averages to sample

    Returns:
        List of (subject_idx, coil_idx, average_idx) tuples
    """
    indices = []
    nifti_list = data.list()

    for subj_idx, nifti in enumerate(nifti_list):
        # Get coil limits
        if 'DIM_COIL' in getattr(nifti, 'dim_tags', []):
            c_max = nifti.shape[nifti.dim_position('DIM_COIL')] - 1
            min_c, max_c = _get_limits(n_coils, c_max)
        else:
            min_c, max_c, c_max = 0, 1, 0

        # Get average limits
        if 'DIM_DYN' in getattr(nifti, 'dim_tags', []):
            a_max = nifti.shape[nifti.dim_position('DIM_DYN')] - 1
            min_a, max_a = _get_limits(n_averages, a_max)
        else:
            min_a, max_a, a_max = 0, 1, 0

        # Generate all combinations
        for n_c in range(min_c, max_c + 1):
            for n_a in range(min_a, max_a + 1):
                indices.append((subj_idx, c_max - n_c, a_max - n_a))

    return indices


def _get_limits(n: Tuple, n_max: int) -> Tuple[int, int]:
    """Convert (min, max) with possible None to concrete limits."""
    min_n = n_max if n[0] is None else max(0, min(n[0], n_max))
    max_n = n_max if n[1] is None else min(n[1], n_max)
    return min_n, max_n


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
    from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus

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
                            batch_size: int):
    """
    Infinite random-sampling generator (on-the-fly mode).

    Each iteration: pick *batch_size* random subjects, sample fresh
    augmentation parameters, apply the pipeline, yield the batch.

    Yields:
        (batch_data, batch_water) — NIfTI_MRS_Plus objects.
    """
    n_subjects = len(data)

    while True:
        indices = [random.randint(0, n_subjects - 1) for _ in range(batch_size)]
        batch_data, batch_water = _make_batch(data, water, indices)
        batch_params = pipeline.sample_batch_parameters(batch_size)
        yield pipeline(batch_data, batch_water, batch_params=batch_params)


def create_fixed_generator(data: NIfTI_MRS_Plus,
                           water: Optional[NIfTI_MRS_Plus],
                           pipeline,
                           batch_size: int,
                           shuffle: bool = False):
    """
    Single-pass generator with fixed augmentation parameters.

    Parameters are sampled **once** at creation and reused for every batch,
    giving consistent results across epochs.

    Yields:
        (batch_data, batch_water) — NIfTI_MRS_Plus objects.
    """
    fixed_params = pipeline.sample_batch_parameters(batch_size)

    n_subjects = len(data)
    indices = list(range(n_subjects))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, n_subjects, batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_data, batch_water = _make_batch(data, water, batch_idx, copy=True)
        yield pipeline(batch_data, batch_water, batch_params=fixed_params)


def create_deterministic_generator(data: NIfTI_MRS_Plus,
                                   water: Optional[NIfTI_MRS_Plus],
                                   pipeline,
                                   batch_size: int,
                                   n_coils: Tuple,
                                   n_averages: Tuple,
                                   shuffle: bool = False):
    """
    Create deterministic generator over all combinations (backend-agnostic).

    DEPRECATED: Use 'fixed' mode instead for simpler iteration.
    This creates a combinatorial explosion of (subject × coils × averages).

    Args:
        data: NIfTI_MRS_Plus with subjects
        water: Optional water reference
        pipeline: Augmentation pipeline to apply
        batch_size: Batch size
        n_coils: (min, max) coils
        n_averages: (min, max) averages
        shuffle: Whether to shuffle indices

    Yields:
        (batch_data, batch_water) tuples
    """
    # Create all possible indices
    indices = create_deterministic_indices(data, n_coils, n_averages)

    if shuffle:
        import random
        random.shuffle(indices)

    # Yield batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data_list, batch_water_list = [], []

        for subj_idx, coil_idx, avg_idx in batch_indices:
            # Get subject
            subj_data = data[subj_idx]
            subj_water = water[subj_idx] if water is not None else None

            # Apply pipeline with specific indices
            aug_data, aug_water = pipeline(
                subj_data, subj_water,
                coil_indices=coil_idx,
                average_indices=avg_idx
            )

            batch_data_list.append(aug_data)
            batch_water_list.append(aug_water)

        yield batch_data_list, batch_water_list


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
                                 framework: str = None):
    """
    Wrap a generator for a specific framework (PyTorch DataLoader, tf.data, etc.).

    Args:
        generator: Generator (iterator) yielding (data, water) batches
        backend: Target backend for data conversion
        framework: 'pytorch', 'tensorflow', 'keras', 'jax', 'numpy', None (raw)

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
