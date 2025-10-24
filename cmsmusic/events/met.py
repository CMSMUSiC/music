import awkward as ak
import uproot
import vector

vector.register_awkward()  # <- important


def _build_met(evts: uproot.TTree, jets: ak.Array) -> ak.Array:
    MET_PREFIX = "PuppiMET_"

    _met = evts.arrays(
        [
            "PuppiMET_pt",
            "PuppiMET_phi",
            "PuppiMET_phiUnclusteredDown",
            "PuppiMET_phiUnclusteredUp",
            "PuppiMET_ptUnclusteredDown",
            "PuppiMET_ptUnclusteredUp",
        ]
    )

    if "PuppiMET_mass" not in ak.fields(_met):
        _met = ak.with_field(
            _met,
            ak.zeros_like(_met["PuppiMET_pt"]),
            "PuppiMET_mass",
        )

    if "PuppiMET_eta" not in ak.fields(_met):
        _met = ak.with_field(
            _met,
            ak.zeros_like(_met["PuppiMET_pt"]),
            "PuppiMET_eta",
        )

    met = ak.zip(
        {f[len(MET_PREFIX) :]: _met[f] for f in ak.fields(_met)},
        with_name="Momentum4D",
    )
    return met
