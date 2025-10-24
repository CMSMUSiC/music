import awkward as ak
import uproot
import vector

vector.register_awkward()  # <- important


def _build_jets(evts: uproot.TTree) -> ak.Array:
    JET_PREFIX = "Jet_"

    _jets = evts.arrays(
        [
            "Jet_pt",
            "Jet_eta",
            "Jet_phi",
            "Jet_mass",
        ]
    )

    jets = ak.zip(
        {f[len(JET_PREFIX) :]: _jets[f] for f in ak.fields(_jets)},
        with_name="Momentum4D",
    )
    return jets
