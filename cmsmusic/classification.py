import logging
from typing import Literal

import awkward as ak

from .dataset import Dataset, DatasetType
from .eras import Year
from .events import Events, EventsBuilder
from .utils import vec, null_vec
from .variation import Variation, VariationEngine, VariationType
from .nb_hist import make_uniform_hist, make_variable_hist, to_hist

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

    variations = [
        Variation(
            name="Nominal",
            variation_type=VariationType.INTEGRAL,
            payload=lambda e: {},
        ),
        Variation(
            name="Lumi_Up",
            variation_type=VariationType.INTEGRAL,
            payload=lambda e: lumi_var(e, "up"),
        ),
        Variation(
            name="Lumi_Down",
            variation_type=VariationType.INTEGRAL,
            payload=lambda e: lumi_var(e, "down"),
        ),
        Variation(
            name="MuonID_Up",
            variation_type=VariationType.CONSTANT,
            payload=lambda e: {"muons": e.data.muons * 10_000},
        ),
    ]

    raw_events = EventsBuilder(dataset, file_index, enable_cache).build()

    for var in variations:
        if dataset.dataset_type == DatasetType.DATA and var.name != "Nominal":
            continue

        with VariationEngine(var, dataset, raw_events) as events:
            print(var, dataset.sum_weights, dataset.num_events)
            events.data.gen_weights.show()
            events.data.muons.pt.show()

            from numba import njit
            import numpy as np

            bar = 123

            @njit
            def foo(data, event_filter):
                h1 = make_uniform_hist(bins=10, low=0.0, high=13600.0, name="regular")
                h2 = make_variable_hist(
                    edges_in=[0.0, 3, 5, 90, 13600], name="non-uniform"
                )
                for idx_evt, evt in enumerate(data):
                    if not event_filter[idx_evt]:
                        continue

                    _sum = 0.0
                    for m in evt.muons:
                        _sum += vec(m).px

                    h1.fill(_sum)
                    h2.fill(_sum)

                print(bar)

                return (h1, h2)

            h1, h2 = foo(events.data, events.get_event_filter())
            print(to_hist(h1), to_hist(h2))

            # here goes the analysis
            # ...

    logger.info(f"Num of events: {raw_events.num_events}")

    return
