import awkward as ak
import uproot
import vector
from .corrections.jet_id import JetId, JetIdWP
from ..datasets import Dataset

from .load_fields import load_fields

vector.register_awkward()  # <- important


def _build_jets(evts: uproot.TTree, dataset: Dataset) -> ak.Array:
    JET_PREFIX = "Jet_"

    _jets = load_fields(
        [
            "Jet_pt",
            "Jet_eta",
            "Jet_phi",
            "Jet_mass",
            "Jet_chHEF",
            "Jet_neHEF",
            "Jet_chEmEF",
            "Jet_neEmEF",
            "Jet_muEF",
            "Jet_chMultiplicity",
            "Jet_neMultiplicity",
        ],
        evts,
    )

    jets = ak.zip(
        {f[len(JET_PREFIX) :]: _jets[f] for f in ak.fields(_jets)},
        with_name="Momentum4D",
    )

    jet_id_tight = JetId(dataset, JetIdWP.AK4PUPPI_Tight)
    jets = ak.with_field(jets, jet_id_tight(jets), "jet_id_tight")

    jet_id_tight_lep_veto = JetId(dataset, JetIdWP.AK4PUPPI_TightLeptonVeto)
    jets = ak.with_field(jets, jet_id_tight_lep_veto(jets), "jet_id_tight_lep_veto")

    return jets
