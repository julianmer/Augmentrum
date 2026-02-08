####################################################################################################
#                                        base_dataset.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-02-07                                                                              #
#                                                                                                  #
# Purpose: Backend-agnostic dataset utilities for Augmentrum.                                      #
#          All main functionality moved to Augmentrum class.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
from typing import List, Optional, Tuple, Dict, Callable, Union

# internal
from augmentrum.core import NIfTI_MRS_Plus, Backend
from augmentrum.sampling.subject_splitter import SubjectSplitter


#**************************************************************************************************#
#                                       Helper Functions                                           #
#**************************************************************************************************#

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


#**************************************************************************************************#
#                                    Backend-Agnostic Generators                                   #
#**************************************************************************************************#

def create_random_generator(data: NIfTI_MRS_Plus,
                            water: Optional[NIfTI_MRS_Plus],
                            pipeline,
                            batch_size: int):
    """
    Create infinite random sampling generator (backend-agnostic).

    Args:
        data: NIfTI_MRS_Plus with subjects
        water: Optional water reference
        pipeline: Augmentation pipeline to apply
        batch_size: Batch size

    Yields:
        (batch_data, batch_water) tuples
    """
    import random
    n_subjects = len(data)

    while True:
        batch_data_list, batch_water_list = [], []

        for _ in range(batch_size):
            # Random subject
            idx = random.randint(0, n_subjects - 1)

            # Get single subject
            subj_data = data[idx]
            subj_water = water[idx] if water is not None else None

            # Apply pipeline (handles random coil/average sampling internally)
            aug_data, aug_water = pipeline(subj_data, subj_water)

            batch_data_list.append(aug_data)
            batch_water_list.append(aug_water)

        yield batch_data_list, batch_water_list


def create_fixed_generator(data: NIfTI_MRS_Plus,
                           water: Optional[NIfTI_MRS_Plus],
                           pipeline,
                           batch_size: int,
                           shuffle: bool = False):
    """
    Create generator with fixed augmentation parameters (backend-agnostic).

    Unlike deterministic mode (which creates all combinations), this simply
    iterates over subjects and applies the pipeline with FIXED parameter values.

    For example:
      - mode='on-the-fly': n_coils=(1,8) → randomly picks 1-8 each time
      - mode='fixed': n_coils=4 → always uses exactly 4 coils

    Args:
        data: NIfTI_MRS_Plus with subjects
        water: Optional water reference
        pipeline: Augmentation pipeline to apply (uses exact values from modules)
        batch_size: Batch size
        shuffle: Whether to shuffle subject order

    Yields:
        (batch_data, batch_water) tuples
    """
    import random

    n_subjects = len(data)
    indices = list(range(n_subjects))

    if shuffle:
        random.shuffle(indices)

    # Yield batches of subjects
    for i in range(0, n_subjects, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data_list, batch_water_list = [], []

        for idx in batch_indices:
            # Get subject
            subj_data = data[idx]
            subj_water = water[idx] if water is not None else None

            # Apply pipeline (modules will use their fixed parameter values)
            aug_data, aug_water = pipeline(subj_data, subj_water)

            batch_data_list.append(aug_data)
            batch_water_list.append(aug_water)

        yield batch_data_list, batch_water_list


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


#**************************************************************************************************#
#                                    Backend Conversion Functions                                  #
#**************************************************************************************************#

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


#**************************************************************************************************#
#                                    Framework DataLoader Wrappers                                 #
#**************************************************************************************************#

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

    # For nifti_list or raw python: return generator directly (no conversion)
    if framework in ['python', 'nifti_list']:
        return generator

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


__all__ = [
    'create_random_generator',
    'create_deterministic_generator',
    'create_deterministic_indices',
    'convert_batch_to_backend',
    'wrap_generator_for_framework',
    'SubjectSplitter',
]
