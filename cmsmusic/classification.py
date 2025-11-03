import logging

from .datasets import Dataset
from .events import Events

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
            f"Processing {dataset.lfns[file_index]} of {dataset.short_str()} ..."
        )

    file_to_process = dataset.lfns[file_index]

    events = Events.build_events(file_to_process, enable_cache)

    if verbose:
        logger.info(events.muons.charge)
        logger.info(events.muons.px)
    return
