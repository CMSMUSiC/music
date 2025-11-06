from typing import NamedTuple
import awkward as ak
import uproot


class Field(NamedTuple):
    name: str
    default: float | bool | int | None = None


def load_fields(fields: list[Field | str], evts: uproot.TTree):
    if len(fields) == 0:
        raise RuntimeError("no fields to load")

    _fields: list[Field] = []
    for f in fields:
        match f:
            case str():
                _fields.append(Field(f))
            case Field():
                _fields.append(f)
            case _:
                raise ValueError("invalid field type")

    fields_to_load: list[Field] = []
    fields_not_found: list[Field] = []
    for f in _fields:
        if f.name not in evts.keys():
            fields_not_found.append(f)
        else:
            fields_to_load.append(f)

    _data = evts.arrays([f.name for f in fields_to_load])

    _fields = ak.fields(_data)
    if len(_fields) == 0:
        raise RuntimeError("no fields loaded")

    template_fields = _data[_fields[0]]
    for f in fields_not_found:
        if f.default is None:
            raise ValueError(f"no default value for {f.name}")

        _data = ak.with_field(
            _data,
            ak.full_like(template_fields, f.default),
            f.name,
        )

    return _data
