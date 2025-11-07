from typing import NamedTuple
import awkward as ak
from numba.core.typing import templates
import numpy as np
import uproot

import logging

logger = logging.getLogger("Events")


class Field(NamedTuple):
    name: str
    default: float | bool | int | None = None
    template: str | None = None


def load_fields(fields: list[Field | str], evts: uproot.TTree):
    if len(fields) == 0:
        raise RuntimeError("no fields to load")
    num_events = evts.num_entries

    _fields: list[Field] = []
    for f in fields:
        match f:
            case str():
                _fields.append(Field(f))
            case Field():
                _fields.append(f)
            case _:
                raise ValueError(f"invalid field type for {f}")

    fields_to_load: list[Field] = []
    fields_not_found: list[Field] = []
    for f in _fields:
        if f.name not in evts.keys():
            fields_not_found.append(f)
        else:
            fields_to_load.append(f)

    _data = evts.arrays([f.name for f in fields_to_load])

    for f in fields_not_found:
        if f.default is None:
            logger.warning(f"No default value for {f.name}. Skipping ...")
            continue

        if f.template is None:
            raise ValueError(f"No template array for {f.name}.")

        if len(_data) != 0:
            _data = ak.with_field(
                _data,
                ak.full_like(evts.arrays([f.template])[f.template], f.default),
                f.name,
            )

        else:
            _data = ak.Array(
                {f.name: ak.full_like(evts.arrays([f.template])[f.template], f.default)}
            )

    return _data
