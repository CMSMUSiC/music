import awkward as ak
import uproot

from .load_fields import load_fields


def _build_flags(evts: uproot.TTree) -> ak.Array:
    FLAG_PREFIX = "Flag_"

    _flag = load_fields(
        [
            "Flag_goodVertices",
            "Flag_globalSuperTightHalo2016Filter",
            "Flag_EcalDeadCellTriggerPrimitiveFilter",
            "Flag_BadPFMuonFilter",
            "Flag_BadPFMuonDzFilter",
            "Flag_hfNoisyHitsFilter",
            "Flag_eeBadScFilter",
            "Flag_ecalBadCalibFilter",
        ],
        evts,
    )

    flag = ak.zip(
        {f[len(FLAG_PREFIX) :]: _flag[f] for f in ak.fields(_flag)},
    )
    return flag
