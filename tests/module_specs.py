####################################################################################################
#                                        module_specs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-01                                                                              #
#                                                                                                  #
# Purpose: One source of truth for the modules the test suite iterates over, discovered from the   #
#          registry rather than hand-listed, so a new module cannot silently escape coverage.      #
#                                                                                                  #
####################################################################################################

"""
Shared module registry for the tests that iterate over modules.

Three tests sweep the whole module set — the backend-compatibility matrix, the
README support table, and the provenance round-trip. Each used to carry its own
hand-written list, so registering a new module in "Augmentrum.AVAILABLE_MODULES"
gave it no coverage anywhere and nothing complained. "KspaceUndersampling" was
in neither list.

Here the *discovery* is automatic and only the per-variant detail is written by
hand. "uncovered_classes" is what closes the gap: a class in the registry
with no spec is a test failure, not silence.

A module may need several specs — "GaussianNoise" has three ways of stating a
noise level, "BaselineAugmentation" three baseline models — so specs are a flat
list of variants, not one per class.
"""


#*************#
#   imports   #
#*************#
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type

from augmentrum.core.augmentrum import Augmentrum
from augmentrum.core.base_module import BaseModule


#**************************************************************************************************#
#                                        Class ModuleSpec                                          #
#**************************************************************************************************#
#                                                                                                  #
# One testable configuration of one module: how to build it and what it needs to run.              #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class ModuleSpec:
    """
    One testable configuration of a module.

    Args:
        label: display name and pytest id, e.g. "GaussianNoise[snr_db]".
        cls: the module class.
        kwargs: constructor arguments for this variant.
        needs_multicoil: needs the multi-coil fixture (DIM_COIL + DIM_DYN).
        spatial: operates on 2-D image tensors "(batch, X, Y, C)" rather than spectra.
        volume: operates on 5-D volumes "(batch, X, Y, Z, T)".
        changes_length: rewrites the spectral length, so shape-preservation
            assertions do not apply — "ZeroFill" and "Apodization[truncate]".
        own_provenance: records its own provenance per operation instead of
            through "self.params", so the constructor-argument check does not
            apply.
        nifti_kwargs: overrides for the NIfTI-list path, where the data layout
            differs from the tensor sweeps'. Only "SpatialAugmentations" needs
            it: the sweeps hand it a 4-D image tensor (dim=2) while a NIfTI list
            is always "(X, Y, Z, T)" (dim=3).
        note: free text surfaced in the README summary for known issues.
    """
    label: str
    cls: Type[BaseModule]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    needs_multicoil: bool = False
    spatial: bool = False
    volume: bool = False
    changes_length: bool = False
    own_provenance: bool = False
    nifti_kwargs: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def build(self) -> BaseModule:
        """A fresh instance of this variant, as the tensor sweeps use it."""
        return self.cls(**self.kwargs)

    def build_for_nifti(self) -> BaseModule:
        """A fresh instance configured for the NIfTI-list path."""
        return self.cls(**{**self.kwargs, **self.nifti_kwargs})

    @property
    def factory(self) -> Callable[[], BaseModule]:
        """Zero-arg factory, for consumers that want a callable."""
        return self.build


#***************#
#   the specs   #
#***************#
# Every class in Augmentrum.AVAILABLE_MODULES needs at least one entry. Add a
# module to the registry without adding one here and test_registry_is_covered
# fails — which is the point of the file.
SPECS: List[ModuleSpec] = [
    ModuleSpec("AmplitudeScaling[uniform]", None, {"scale_factor": 0.8}),
    ModuleSpec("AmplitudeScaling[range]", None, {"scale_factor": (0.7, 1.3)}),

    ModuleSpec("Apodization[exponential]", None, {"mode": "exponential", "lb_hz": 5.0}),
    ModuleSpec("Apodization[truncate]", None, {"mode": "truncate", "n_pts": 256},
               changes_length=True),

    ModuleSpec("ArtificialPeaks", None, {}),

    ModuleSpec("BaselineAugmentation[random_walk]", None, {"mode": "random_walk"}),
    ModuleSpec("BaselineAugmentation[bspline]", None, {"mode": "bspline"},
               note="float32 precision -> bspline solver NaN/overflow (TODO)"),
    ModuleSpec("BaselineAugmentation[polynomial]", None, {"mode": "polynomial"}),

    ModuleSpec("AverageSampler", None, {"mode": "random"}, needs_multicoil=True),

    ModuleSpec("CoilSampler[draw]", None, {"mode": "random"}, needs_multicoil=True),

    ModuleSpec("EddyCurrent[synthetic]", None, {"mode": "synthetic"}),

    ModuleSpec("FrequencyShift", None, {"shift_hz": 5.0}),

    ModuleSpec("GaussianNoise[sigma]", None, {"sigma": 0.01}),
    ModuleSpec("GaussianNoise[snr_db]", None, {"snr_db": 20.0}),
    ModuleSpec("GaussianNoise[sigma_frac]", None, {"sigma_frac": 0.02}),

    ModuleSpec("CoilSampler[synthesize]", None,
               {"mode": "synthesize", "n_coils": 4, "seed": 0}, volume=True),

    ModuleSpec("KspaceUndersampling[cartesian]", None,
               {"ksp_mode": "cartesian", "acceleration_factor": 2.0, "us_seed": 0},
               volume=True),
    ModuleSpec("KspaceUndersampling[gridded]", None,
               {"ksp_mode": "gridded", "trajectory": "radial_2d",
                "acceleration_factor": 2.0, "us_seed": 0}, volume=True),

    ModuleSpec("LineBroadening[lorentzian]", None, {"lb_hz": 5.0, "mode": "lorentzian"}),
    ModuleSpec("LineBroadening[gaussian]", None, {"gb_hz": 5.0, "mode": "gaussian"}),
    ModuleSpec("LineBroadening[voigt]", None, {"lb_hz": 3.0, "gb_hz": 2.0, "mode": "voigt"}),

    # Every operation it switches on writes its own FSL-MRS provenance entry, so
    # 'Coil combination' being present IS coil=True and its absence is coil=False.
    # Recording the constructor arguments as well would duplicate that, and would
    # state the intent rather than what ran.
    ModuleSpec("NIfTI_RawProcessor", None, {}, needs_multicoil=True,
               own_provenance=True),

    ModuleSpec("PhaseShift[zero_order]", None, {"zero_order_deg": 30.0}),
    ModuleSpec("PhaseShift[first_order]", None, {"first_order_deg": 45.0}),

    ModuleSpec("ResidualWater", None, {}),

    # dim=2: the module sweeps build a 4-D (batch, X, Y, C) tensor.
    ModuleSpec("SpatialAugmentations", None, {"dim": 2, "prob": 1.0}, spatial=True,
               nifti_kwargs={"dim": 3}),

    ModuleSpec("SpuriousEchoes[replica]", None, {"mode": "replica"}),
    ModuleSpec("SpuriousEchoes[hybrid]", None, {"mode": "hybrid"}),

    ModuleSpec("ZeroFill[pad]", None, {"target_pts": 1024}, changes_length=True),
    ModuleSpec("ZeroFill[crop]", None, {"target_pts": 256}, changes_length=True),
]


#***************#
#   discovery   #
#***************#
def registered_classes() -> Dict[str, Type[BaseModule]]:
    """Distinct module classes reachable through "Augmentrum.AVAILABLE_MODULES"."""
    return {c.__name__: c for c in Augmentrum.AVAILABLE_MODULES.values()}


def specs_for(name: str) -> List[ModuleSpec]:
    """Every spec whose label names class *name*."""
    return [s for s in SPECS if s.label.split('[')[0] == name]


def uncovered_classes() -> List[str]:
    """Registered classes with no spec — the gap this file exists to close."""
    covered = {s.label.split('[')[0] for s in SPECS}
    return sorted(set(registered_classes()) - covered)


def _bind_classes() -> None:
    """
    Resolve each spec's "cls" from its label against the live registry.

    Specs name their class in the label rather than importing it, so this file
    has one import of "Augmentrum" and no second list of imports to drift out
    of step with the registry.
    """
    known = registered_classes()
    for i, spec in enumerate(SPECS):
        name = spec.label.split('[')[0]
        if name not in known:
            raise KeyError(
                f"SPECS[{i}] names {name!r}, which is not in "
                f"Augmentrum.AVAILABLE_MODULES. Known: {sorted(known)}"
            )
        SPECS[i] = ModuleSpec(spec.label, known[name], spec.kwargs,
                              spec.needs_multicoil, spec.spatial, spec.volume,
                              spec.changes_length, spec.own_provenance,
                              spec.nifti_kwargs, spec.note)


_bind_classes()
