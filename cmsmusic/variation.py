from collections.abc import Callable
from enum import IntEnum, auto
from typing import NamedTuple

import awkward as ak

from .dataset import Dataset, DatasetType
from .events import Events


class VariationType(IntEnum):
    INTEGRAL = auto()
    CONSTANT = auto()
    DIFFERENTIAL = auto()


type DataModifier = Callable[[Events], dict[str, ak.Array]]


class Variation(NamedTuple):
    name: str
    variation_type: VariationType
    payload: DataModifier


class Variations:
    def __init__(self, variations: list[Variation]) -> None:
        self.variations = variations
        if self.variations[0].name != "Nominal":
            raise ValueError('the first variation should be "Nominal"')

    @property
    def constants(self) -> list[Variation]:
        return [
            v for v in self.variations if v.variation_type == VariationType.CONSTANT
        ]

    @property
    def integrals(self) -> list[Variation]:
        return [
            v for v in self.variations if v.variation_type == VariationType.INTEGRAL
        ]

    @property
    def differentials(self) -> list[Variation]:
        return [
            v for v in self.variations if v.variation_type == VariationType.DIFFERENTIAL
        ]


class VariationEngine:
    def __init__(self, variation: Variation, dataset: Dataset, events: Events):
        if dataset.dataset_type == DatasetType.DATA:
            self.skip = True
            return

        self.variation = variation
        self.events = events
        self.buffer: dict[str, ak.Array]
        self.payload = self.variation.payload(self.events, dataset)
        self.skip = False

    def __enter__(self):
        if self.skip:
            return self.events

        for field in self.payload:
            # buffer old field
            self.events.data[f"{field}_OLD"] = self.events.data[field]

            # set new data
            self.events.data[field] = self.payload[field]

        return self.events

    def __exit__(self, exc_type, exc_value, traceback):
        if self.skip:
            return

        for field in self.payload:
            # recover buffered field
            self.events.data[field] = self.events.data[f"{field}_OLD"]

            # drop useless data
            del self.events.data[f"{field}_OLD"]
