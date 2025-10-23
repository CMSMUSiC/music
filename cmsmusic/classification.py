from .datasets import Dataset
from .events import Events


def run_classification(
    file_index: int, dataset: Dataset, silence_mode: bool, enable_cache: bool
) -> None:
    """
    Will classify one file
    """
    assert isinstance(dataset.lfns, list)
    if file_index >= len(dataset.lfns):
        raise IndexError(
            f"{file_index} is larger then the length of the {dataset.short_str()} ({len(dataset.lfns)} files)"
        )

    if not silence_mode:
        print(f"Processing {dataset.lfns[file_index]} of {dataset.short_str()} ...")

    file_to_process = dataset.lfns[file_index]

    events = Events.build_events(file_to_process, enable_cache)

    if not silence_mode:
        print(events.muons.charge)
        print(events.muons.px)
    return
