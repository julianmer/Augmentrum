####################################################################################################
#                                    test_module_registry.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-01                                                                              #
#                                                                                                  #
# Purpose: Guards that every registered module is covered by the shared spec table, and that each  #
#          one records its constructor parameters in provenance.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import inspect

import pytest

from augmentrum.core.augmentrum import Augmentrum
from augmentrum.core.base_module import BaseModule
from tests.module_specs import SPECS, registered_classes, uncovered_classes


#*************#
#   the gap   #
#*************#

def test_every_registered_module_has_a_spec():
    """
    A module in the registry with no spec gets no coverage from any sweeping test.

    This is the guard the suite was missing. The backend-compatibility matrix and
    the README support table both used to carry their own hand-written lists, so
    registering a module gave it coverage nowhere and nothing failed —
    ``KspaceUndersampling`` was in neither, and ``AmplitudeScaling`` was not even
    reachable by name.
    """
    missing = uncovered_classes()
    assert not missing, (
        f"registered in Augmentrum.AVAILABLE_MODULES but absent from SPECS: {missing}. "
        f"Add a ModuleSpec in tests/module_specs.py."
    )


def test_specs_name_only_registered_classes():
    """The reverse direction: a spec for something unregistered is a stale spec."""
    known = set(registered_classes())
    named = {s.label.split('[')[0] for s in SPECS}
    assert named <= known, f"SPECS names classes not in the registry: {sorted(named - known)}"


def test_every_spec_constructs():
    """Every variant must be buildable, or the sweeps below silently skip it."""
    for spec in SPECS:
        assert isinstance(spec.build(), BaseModule), spec.label


#****************#
#   provenance   #
#****************#

@pytest.mark.parametrize("spec", SPECS, ids=[s.label for s in SPECS])
def test_constructor_params_reach_provenance(spec):
    """
    Every constructor argument a module accepts must land in ``self.params``.

    ``self.params`` is what ``BaseModule`` embeds in ``operation_details`` and
    therefore what reaches the NIfTI header. A module that forwards only some of
    its arguments to ``super().__init__`` produces a dataset whose provenance
    does not say what the module was configured to do —
    ``SpatialAugmentations`` forwarded five of nineteen, so no run recorded which
    zoom or rotation range was permitted.
    """
    if spec.own_provenance:
        pytest.skip(f"{spec.label} records provenance per operation, not via self.params")

    module = spec.build()
    accepted = {
        p.name for p in inspect.signature(spec.cls.__init__).parameters.values()
        if p.name not in ('self', 'args', 'kwargs')
    }
    recorded = set(module.params)
    missing = sorted(accepted - recorded)
    assert not missing, (
        f"{spec.label}: accepts {sorted(accepted)} but records {sorted(recorded)}; "
        f"missing from provenance: {missing}. Forward them to super().__init__()."
    )


@pytest.mark.parametrize("spec", SPECS, ids=[s.label for s in SPECS])
def test_explicit_kwargs_are_recorded_by_value(spec):
    """The values passed in, not just the keys, must survive into provenance."""
    module = spec.build()
    for key, value in spec.kwargs.items():
        assert key in module.params, f"{spec.label}: {key!r} absent from params"
        assert module.params[key] == value, (
            f"{spec.label}: params[{key!r}] is {module.params[key]!r}, expected {value!r}"
        )


#**************#
#   volatile   #
#**************#

def test_volatile_skips_metadata_updates():
    """
    ``volatile=True`` is documented as skipping metadata updates for speed.

    Provenance is metadata, so it is skipped too — the trade being bought. Pinned
    here so the meaning of the flag cannot drift silently.
    """
    from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus

    source = inspect.getsource(NIfTI_MRS_Plus)
    assert 'self.volatile' in source, "NIfTI_MRS_Plus no longer honours volatile"

    plus = NIfTI_MRS_Plus(nifti_list=[], volatile=True)
    assert plus.volatile is True
    assert NIfTI_MRS_Plus(nifti_list=[], volatile=False).volatile is False


#**************#
#   registry   #
#**************#

def test_registry_aliases_resolve_to_module_classes():
    """Every name a user can put in a pipeline must map to a BaseModule subclass."""
    for name, cls in Augmentrum.AVAILABLE_MODULES.items():
        assert isinstance(cls, type) and issubclass(cls, BaseModule), (
            f"AVAILABLE_MODULES[{name!r}] is {cls!r}, not a BaseModule subclass"
        )


def test_exported_augmentations_are_reachable_by_name():
    """
    Anything exported from ``augmentrum.augmentation`` should be usable in a pipeline.

    ``AmplitudeScaling`` was exported, documented in the README table and covered
    by the backend tests, yet absent from the registry — so it could not be named
    in a pipeline at all.
    """
    import augmentrum.augmentation as pkg

    registered = set(registered_classes())
    exported = {
        name for name in pkg.__all__
        if isinstance(getattr(pkg, name), type)
        and issubclass(getattr(pkg, name), BaseModule)
    }
    assert exported <= registered, (
        f"exported but not registered in AVAILABLE_MODULES: {sorted(exported - registered)}"
    )


#***************************#
#   provenance end to end   #
#***************************#

def _nifti_list(shape, n=2):
    """*n* NIfTI-MRS objects of the given ``(X, Y, Z, T)`` shape."""
    import numpy as np
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    rng = np.random.default_rng(0)
    return [
        gen_nifti_mrs(
            (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64'),
            1 / 2000.0, 123.0,
        )
        for _ in range(n)
    ]


def _processing_applied(nifti):
    try:
        return nifti.hdr_ext['ProcessingApplied']
    except (KeyError, TypeError):
        return []


@pytest.mark.parametrize("spec", SPECS, ids=[s.label for s in SPECS])
def test_running_a_module_writes_provenance(spec):
    """
    Running a module must leave a record in both provenance stores.

    ``self.params`` being populated is necessary but not sufficient — the record
    has to survive the call and reach the NIfTI header extension, which is what a
    downstream reader actually sees. Checked per module rather than once, because
    the four data paths (spectra, multi-coil, volume, image) route through
    different branches of BaseModule.
    """
    from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend

    shape = (8, 8, 4, 16) if spec.volume or spec.nifti_kwargs else (1, 1, 1, 512)
    plus = NIfTI_MRS_Plus(nifti_list=_nifti_list(shape),
                          backend=Backend.NIFTI_LIST, volatile=False)
    out, _ = spec.build_for_nifti()(plus, None)

    common = out.metadata_common.get('common_provenance', [])
    assert common, f"{spec.label}: nothing recorded in common_provenance"

    applied = _processing_applied(out.nifti_list[0])
    assert applied, f"{spec.label}: nothing reached the NIfTI ProcessingApplied header"

    if spec.own_provenance:
        # Its operations each write their own entry, so the trail is richer than
        # the single module-level record — that is the point of the exemption.
        assert len(applied) > 1, (
            f"{spec.label} claims own_provenance but wrote only {len(applied)} entry"
        )


@pytest.mark.parametrize("spec", SPECS, ids=[s.label for s in SPECS])
def test_volatile_writes_no_provenance(spec):
    """``volatile=True`` buys speed by skipping exactly this bookkeeping."""
    from augmentrum.core.nifti_mrs_plus import NIfTI_MRS_Plus, Backend

    shape = (8, 8, 4, 16) if spec.volume or spec.nifti_kwargs else (1, 1, 1, 512)
    plus = NIfTI_MRS_Plus(nifti_list=_nifti_list(shape),
                          backend=Backend.NIFTI_LIST, volatile=True)
    out, _ = spec.build_for_nifti()(plus, None)
    assert not out.metadata_common.get('common_provenance', []), (
        f"{spec.label}: wrote provenance despite volatile=True"
    )
