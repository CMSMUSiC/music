import logging
from typing import Literal

import awkward as ak

from .dataset import Dataset
from .eras import Year
from .events import Events, EventsBuilder

# from .systematics import systematics
from .variation import Variation, VariationEngine, Variations, VariationType

logger = logging.getLogger("Classification")


def run_classification(file_index: int, dataset: Dataset, enable_cache: bool) -> None:
    """
    Classify one file
    """
    assert isinstance(dataset.lfns, list)
    if file_index >= len(dataset.lfns):
        raise IndexError(
            f"{file_index} is larger then the length of the {dataset.short_str()} ({len(dataset.lfns)} files)"
        )

    logger.info(f"Processing {dataset.lfns[file_index]} from {dataset.short_str()} ...")

    def lumi_var(
        events: Events, shift: Literal["up"] | Literal["down"]
    ) -> dict[str, ak.Array]:
        mult_factor = 1.0
        if shift == "down":
            mult_factor = -1.0

        match dataset.year:
            case Year.RunSummer24:
                uncert = 1.3 / 100.0
            case Year.RunSummer23:
                raise NotImplementedError(dataset.year)
            case Year.RunSummer22EE:
                raise NotImplementedError(dataset.year)
            case Year.RunSummer22:
                raise NotImplementedError(dataset.year)
            case Year.Run2018:
                raise NotImplementedError(dataset.year)
            case Year.Run2017:
                raise NotImplementedError(dataset.year)
            case Year.Run2016preVFP:
                raise NotImplementedError(dataset.year)
            case Year.Run2016postVFP:
                raise NotImplementedError(dataset.year)
            case _:
                raise ValueError(f"Invalid year {dataset.year}")

        return {"int_lumi": events.data.int_lumi * (1 + mult_factor * uncert / 100)}

    systematics = [
        Variation(
            name="Nominal",
            variation_type=VariationType.INTEGRAL,
            payload=lambda e: {},
        ),
        Variation(
            name="Lumi_Up",
            variation_type=VariationType.INTEGRAL,
            payload=lambda e: lumi_var(e, d, "up"),
        ),
        Variation(
            name="Lumi_Down",
            variation_type=VariationType.INTEGRAL,
            payload=lambda e: lumi_var(e, d, "down"),
        ),
        Variation(
            name="MuonID_Up",
            variation_type=VariationType.CONSTANT,
            payload=lambda e: {"muons": e.data.muons * 10_000},
        ),
    ]

    raw_events = EventsBuilder(dataset, file_index, enable_cache).build()

    variations = Variations(systematics)

    for var in variations.constants:
        with VariationEngine(var, dataset, raw_events) as events:
            print(events.data.muons)
            # here goes the analysis

    logger.info(f"Num of events: {raw_events.num_events}")

    return
