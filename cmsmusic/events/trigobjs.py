import awkward as ak
import uproot
import vector

from .load_fields import Field, load_fields

vector.register_awkward()  # <- important


def _build_trigobjs(evts: uproot.TTree) -> ak.Array:
    TRIGOBJS_PREFIX = "TrigObj_"

    _trigobjs = load_fields(
        [
            "TrigObj_eta",
            "TrigObj_filterBits",
            "TrigObj_id",
            "TrigObj_phi",
            "TrigObj_pt",
            Field("TrigObj_mass", 0.0),
        ],
        evts,
    )

    trigobjs = ak.zip(
        {f[len(TRIGOBJS_PREFIX) :]: _trigobjs[f] for f in ak.fields(_trigobjs)},
        with_name="Momentum4D",
    )
    return trigobjs
