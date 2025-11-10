import awkward as ak
import uproot
import vector

from .load_fields import Field, load_fields

vector.register_awkward()  # <- important


MUON_MASS = 0.105_658_374_5


def _build_muons(evts: uproot.TTree) -> ak.Array:
    MUON_PREFIX = "Muon_"

    _muons = load_fields(
        [
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            Field("Muon_mass", MUON_MASS),
            "Muon_charge",
            "Muon_isPFcand",
        ],
        evts,
    )

    muons = ak.zip(
        {f[len(MUON_PREFIX) :]: _muons[f] for f in ak.fields(_muons)},
        with_name="Momentum4D",
    )

    return muons
