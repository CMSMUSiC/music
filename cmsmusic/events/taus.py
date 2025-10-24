import awkward as ak
import uproot
import vector

vector.register_awkward()  # <- important


def _build_taus(evts: uproot.TTree) -> ak.Array:
    TAU_PREFIX = "Tau_"

    _taus = evts.arrays(
        [
            "Tau_pt",
            "Tau_eta",
            "Tau_phi",
            "Tau_mass",
            "Tau_charge",
        ]
    )

    if "Tau_mass" not in ak.fields(_taus):
        TAU_MASS = 1.7769
        _taus = ak.with_field(
            _taus,
            ak.ones_like(_taus["Tau_pt"]) * TAU_MASS,
            "Tau_mass",
        )

    taus = ak.zip(
        {f[len(TAU_PREFIX) :]: _taus[f] for f in ak.fields(_taus)},
        with_name="Momentum4D",
    )

    return taus
