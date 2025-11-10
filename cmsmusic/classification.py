import logging
from typing import Literal
import gc

import awkward as ak
from numba import njit

from .dataset import Dataset, DatasetType
from .eras import Year
from .events import Events, EventsBuilder
from .utils import vec, null_vec
from .variation import Variation, VariationEngine, VariationType
from .nb_hist import make_uniform_hist, make_variable_hist, to_hist, to_root

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

    # define events tranformers
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

    # variations to run
    variations = [
        Variation(
            name="Nominal",
            variation_type=VariationType.INTEGRAL,
            transformer=lambda e: {},
        ),
        Variation(
            name="Lumi_Up",
            variation_type=VariationType.INTEGRAL,
            transformer=lambda e: lumi_var(e, "up"),
        ),
        Variation(
            name="MuonID_Up",
            variation_type=VariationType.CONSTANT,
            transformer=lambda e: {"muons": e.data.muons * 10_000.0},
        ),
        Variation(
            name="MuonID_Down",
            variation_type=VariationType.CONSTANT,
            transformer=lambda e: {"muons": e.data.muons / 10_000.0},
        ),
        Variation(
            name="Lumi_Down",
            variation_type=VariationType.INTEGRAL,
            transformer=lambda e: lumi_var(e, "down"),
        ),
    ]

    # ensure no repeated variations
    assert len(variations) == len(
        set([v.name for v in variations])
    ), "There are repeated variations"

    # load and build event data
    def apply_nominal_corrections(events):
        logger.warning("TODO: implement nominal corrections")
        # events.data.muons["pt"] = events.data.muons.pt * 10e5

        return events

    nominal_events = (
        EventsBuilder(dataset, file_index, enable_cache)
        .add_transformation(apply_nominal_corrections)
        .build()
    )

    @njit
    def do_classification(data, event_filter):
        h = make_uniform_hist(bins=30, low=70.0, high=110.0, name="regular")
        for idx_evt, evt in enumerate(data):
            if not event_filter[idx_evt]:
                continue

            if not evt.hlt_bits.HLT_IsoMu24:
                continue

            for i, m1 in enumerate(evt.muons):
                for j, m2 in enumerate(evt.muons):
                    if j > i:
                        if m1.pt > 7.0 and m2.pt > 7.0:
                            m1 = vec(m1)
                            m2 = vec(m2)
                            z_cand = m1 + m2
                            if 70 <= z_cand.mass <= 110.0:
                                h.fill(z_cand.mass)

        return h

    for var in variations:
        if dataset.dataset_type == DatasetType.DATA and var.name != "Nominal":
            continue

        with VariationEngine(var, dataset, nominal_events) as events:
            # here goes the analysis ...

            h = do_classification(events.data, events.get_event_filter())

            root_hist = to_root(h)
            root_hist.Print("all")
            print(to_hist(h))
            # print(to_hist(h1), to_hist(h2))

    logger.info(f"Num of events: {nominal_events.num_events}")

    return
