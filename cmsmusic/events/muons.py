import awkward as ak
import uproot
import vector

vector.register_awkward()  # <- important


def _add_muon_corr(muons: ak.Array) -> ak.Array:
    SHIFT = 0.15

    muons = ak.with_field(muons, (muons * (1.0 + SHIFT)).pt, "muons_pt_up")
    muons = ak.with_field(muons, (muons * (1.0 + SHIFT)).mass, "muons_mass_up")

    muons = ak.with_field(muons, (muons * (1.0 - SHIFT)).pt, "muons_pt_down")
    muons = ak.with_field(muons, (muons * (1.0 - SHIFT)).mass, "muons_mass_down")

    return muons


def _build_muons(evts: uproot.TTree) -> ak.Array:
    MUON_PREFIX = "Muon_"

    _muons = evts.arrays(
        [
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
            "Muon_isPFcand",
        ]
    )

    if "Muon_mass" not in ak.fields(_muons):
        MUON_MASS = 0.105_658_374_5
        _muons = ak.with_field(
            _muons, ak.ones_like(_muons["Muon_pt"]) * MUON_MASS, "Muon_mass"
        )

    muons = ak.zip(
        {f[len(MUON_PREFIX) :]: _muons[f] for f in ak.fields(_muons)},
        with_name="Momentum4D",
    )

    muons = _add_muon_corr(muons)

    return muons
