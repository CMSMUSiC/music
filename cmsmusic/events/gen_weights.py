import awkward as ak
import uproot

from .load_fields import Field, load_fields


def _build_gen_weights(evts: uproot.TTree) -> ak.Array:
    gen_weights = load_fields(
        [
            Field("genWeight", 1.0, "run"),
            Field("LHEWeight_originalXWGTUP", 1.0, "run"),
        ],
        evts,
    )

    return gen_weights
