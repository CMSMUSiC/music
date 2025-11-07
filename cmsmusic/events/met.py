import awkward as ak
import uproot
import vector

from .load_fields import Field, load_fields

vector.register_awkward()  # <- important


def _build_met(evts: uproot.TTree, jets: ak.Array) -> ak.Array:
    MET_PREFIX = "PuppiMET_"

    _met = load_fields(
        [
            "PuppiMET_pt",
            "PuppiMET_phi",
            "PuppiMET_phiUnclusteredDown",
            "PuppiMET_phiUnclusteredUp",
            "PuppiMET_ptUnclusteredDown",
            "PuppiMET_ptUnclusteredUp",
            Field("PuppiMET_mass", 0.0, "PuppiMET_pt"),
            Field("PuppiMET_eta", 0.0, "PuppiMET_pt"),
        ],
        evts,
    )

    met = ak.zip(
        {f[len(MET_PREFIX) :]: _met[f] for f in ak.fields(_met)},
        with_name="Momentum4D",
    )
    return met
