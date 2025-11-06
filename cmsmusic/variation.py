from collections.abc import Callable
from enum import Enum, IntEnum, auto
from .events import Events
import awkward as ak
from typing import NamedTuple, Self, Iterable

type DataModifier = Callable[[Events], dict[str, ak.Array]]


class VariationType(IntEnum):
    INTEGRAL = auto()
    CONSTANT = auto()
    DIFFERENTIAL = auto()


class Variation(NamedTuple):
    name: str
    variation_type: VariationType
    payload: DataModifier


class Variations:
    def __init__(self, variations: list[Variation]) -> None:
        self.variations = variations

    def constants(self) -> list[Variation]:
        return [
            v for v in self.variations if v.variation_type == VariationType.CONSTANT
        ]

    def integrals(self) -> list[Variation]:
        return [
            v for v in self.variations if v.variation_type == VariationType.INTEGRAL
        ]

    def differentials(self) -> list[Variation]:
        return [
            v for v in self.variations if v.variation_type == VariationType.DIFFERENTIAL
        ]


class VariationEngine:
    def __init__(self, variation: Variation, events: Events):
        self.variation = variation
        self.events = events

    def __enter__(self):
        return self.events

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Closing {self.path!r}")
        self.file.close()
        if exc_type:
            print(f"Exception: {exc_value}")
        return False  # propagate exception


vars = [
    Variation(
        name="Nominal",
        variation_type=VariationType.INTEGRAL,
        payload=lambda _: {},
    ),
    Variation(
        name="MuonIDUp",
        variation_type=VariationType.INTEGRAL,
        payload=lambda e: {"muons.pt": e.data.muons.pt * 999.0},
    ),
]
