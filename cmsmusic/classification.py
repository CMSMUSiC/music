import logging

from .datasets import Dataset
from .events import EventsBuilder

logger = logging.getLogger("Classification")


def run_classification(
    file_index: int, dataset: Dataset, verbose: bool, enable_cache: bool
) -> None:
    """
    Will classify one file
    """
    assert isinstance(dataset.lfns, list)
    if file_index >= len(dataset.lfns):
        raise IndexError(
            f"{file_index} is larger then the length of the {dataset.short_str()} ({len(dataset.lfns)} files)"
        )

    if verbose:
        logger.info(
            f"Processing {dataset.lfns[file_index]} from {dataset.short_str()} ..."
        )

    events = EventsBuilder(dataset, file_index, enable_cache).build()

    if verbose:
        logger.info(events.data.muons.charge)
        logger.info(events.data.muons.px)
        logger.info(f"Num of events: {events.num_events}")

    return
