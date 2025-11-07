import awkward as ak
import uproot

from ..dataset import Dataset
from ..eras import Year


def _build_int_lumi(evts: uproot.TTree, runs, dataset: Dataset) -> ak.Array:
    match dataset.year:
        case Year.RunSummer24:
            # From: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#DATA_AN1
            int_lumi = ak.Array([109.09] * evts.num_entries)
        case Year.RunSummer23BPix:
            raise NotImplementedError(dataset.year)
        case Year.RunSummer23:
            raise NotImplementedError(dataset.year)
        case Year.RunSummer22EE:
            raise NotImplementedError(dataset.year)
        case Year.RunSummer22:
            raise NotImplementedError(dataset.year)
        case Year.Run2018:
            raise NotImplementedError(dataset.year)
        case Year.Run2017:
            raise NotImplementedError(dataset.year)
        case Year.Run2016preVFP:
            raise NotImplementedError(dataset.year)
        case Year.Run2016postVFP:
            raise NotImplementedError(dataset.year)
        case _:
            raise ValueError(f"Invalid year {year}")

    return int_lumi
