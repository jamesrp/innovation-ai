"""Frozen shared test support for effect, dogma, and card suites.

This package exists so 105 card agents do not write 105 incompatible fixtures. It is owned by the
effect-contract owner; card agents use it and never build states by hand-mutating frozen
dataclasses.

``scenario`` builds an arbitrary validated mid-game position, ``resolve_dogma`` drives one whole
dogma action from a scripted list of choices, and ``assert_resumes_identically`` proves that every
decision boundary round-trips through the versioned state schema to an identical hash.
"""

from .assertions import (
    assert_conserved,
    assert_no_leak,
    assert_resumes_identically,
)
from .scenario import (
    DogmaResult,
    ScenarioBuilder,
    choose_branch,
    choose_card,
    choose_color,
    choose_player,
    choose_splay,
    choose_value,
    decline,
    finish,
    resolve_dogma,
    scenario,
)

__all__ = [
    "DogmaResult",
    "ScenarioBuilder",
    "assert_conserved",
    "assert_no_leak",
    "assert_resumes_identically",
    "choose_branch",
    "choose_card",
    "choose_color",
    "choose_player",
    "choose_splay",
    "choose_value",
    "decline",
    "finish",
    "resolve_dogma",
    "scenario",
]
