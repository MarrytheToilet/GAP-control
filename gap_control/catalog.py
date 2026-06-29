"""Atomic attribute catalog for compositional control.

An *atomic* is a single controllable attribute (one Component) that we cache a per-attribute
advantage for. Any control condition is then a weighted subset of atomics, and — because the
advantage is exactly linear in the reward mixture — its teacher target is the weighted sum of
the cached atomic advantages (see teacher.estimate_prefix_multi).

The catalog is read from the config `atomics` list so soft/hard mixes are configurable.
Each atomic has a stable string id ("sentiment:positive", "length:short", "keyword:ocean")
and maps to a control slot (attributes.SLOTS); keyword atomics share the keyword:present slot
but carry their own surface string.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .attributes import Component, ControlCondition, SLOTS, dim_of


@dataclass
class Atomic:
    id: str
    component: Component
    slot: int
    keywords: Optional[List[str]] = None

    def slot_value(self):
        return self.component.value


def _slot_for(comp: Component) -> int:
    if comp.dim == "keyword":
        return SLOTS[("keyword", "present")]
    if comp.dim == "length" and comp.length_target is not None:
        from .verifiers import LENGTH_BUCKETS
        bucket = next(b for b, (lo, hi) in LENGTH_BUCKETS.items()
                      if lo <= comp.length_target <= hi)
        return SLOTS[("length", bucket)]
    return SLOTS[(comp.dim, comp.value)]


def atomic_id(spec: dict) -> str:
    if spec["dim"] == "keyword":
        return "keyword:" + "_".join(spec.get("keywords", []))
    return f"{spec['dim']}:{spec.get('value')}"


def parse_atomics(specs: List[dict]) -> List[Atomic]:
    atoms = []
    for spec in specs:
        comp = Component(**spec)
        atoms.append(Atomic(id=atomic_id(spec), component=comp,
                            slot=_slot_for(comp), keywords=spec.get("keywords")))
    return atoms


def condition_from_atomics(atoms: List[Atomic], intensities: List[float],
                           aggregate: str = "mean", conflict_mu: float = 0.0) -> ControlCondition:
    comps = []
    for a, w in zip(atoms, intensities):
        c = Component(**vars(a.component))
        c.alpha = w
        comps.append(c)
    return ControlCondition(comps, aggregate=aggregate, conflict_mu=conflict_mu)
