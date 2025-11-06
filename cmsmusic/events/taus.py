import awkward as ak
import uproot
import vector

from .load_fields import Field, load_fields

vector.register_awkward()  # <- important


TAU_MASS = 1.7769


def _build_taus(evts: uproot.TTree) -> ak.Array:
    TAU_PREFIX = "Tau_"

    _taus = load_fields(
        [
            "Tau_pt",
            "Tau_eta",
            "Tau_phi",
            Field("Tau_mass", TAU_MASS),
            "Tau_charge",
        ],
        evts,
    )

    taus = ak.zip(
        {f[len(TAU_PREFIX) :]: _taus[f] for f in ak.fields(_taus)},
        with_name="Momentum4D",
    )

    return taus
