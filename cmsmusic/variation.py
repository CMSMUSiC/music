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


class VariationEngine:
    def __init__(self, variation: Variation, dataset: Dataset, events: Events):
        self.variation = variation
        self.events = events
        self.buffer: dict[str, ak.Array]
        self.payload = self.variation.payload(self.events)
        self.skip = False

    def __enter__(self):
        for field in self.payload:
            # buffer old field
            self.events.data[f"{field}_OLD"] = self.events.data[field]

            # set new data
            self.events.data[field] = self.payload[field]

        return self.events

    def __exit__(self, exc_type, exc_value, traceback):
        for field in self.payload:
            # recover buffered field
            self.events.data[field] = self.events.data[f"{field}_OLD"]

            # drop useless data
            del self.events.data[f"{field}_OLD"]
