import correctionlib
import awkward as ak
import numpy as np

from cmsmusic.ak_utils import flat_np_view

from ...eras import Year, LHCRun
from ...ak_utils import *
from ...datasets import Dataset


class JetVetoMaps:
    def __init__(self, dataset: Dataset) -> None:
        self.year = dataset.year
        self.lhc_run = dataset.lhc_run

        match self.year:
            case Year.RunSummer24:
                self.evaluator = correctionlib.CorrectionSet.from_file(
                    "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/jetvetomaps.json.gz"
                )["Summer24Prompt24_RunBCDEFGHI_V1"]
            case Year.RunSummer23BPix:
                raise NotImplementedError(self.year)
            case Year.RunSummer23:
                raise NotImplementedError(self.year)
            case Year.RunSummer22EE:
                raise NotImplementedError(self.year)
            case Year.RunSummer22:
                raise NotImplementedError(self.year)
            case Year.Run2018:
                raise NotImplementedError(self.year)
            case Year.Run2017:
                raise NotImplementedError(self.year)
            case Year.Run2016preVFP:
                raise NotImplementedError(self.year)
            case Year.Run2016postVFP:
                raise NotImplementedError(self.year)
            case _:
                raise ValueError(f"Invalid year {year}")

    def __call__(self, jets, muons) -> ak.Array:
        muons = muons[muons.isPFcand]

        # loose jet selection
        jets_mask = (
            (jets.pt > 15.0)
            & (jets.jet_id_tight == 1)
            & (jets.chEmEF < 0.9)
            & ak.all(deltaR_table(jets, muons) >= 0.2, axis=-1)
        )

        if self.lhc_run == LHCRun.Run2:
            jets_pu_mask = (jets.puId >= 4) | (jets.pt >= 50.0)
            jets_mask = jets_mask & jets_pu_mask

        jets_eta = flat_np_view(jets.eta[jets_mask])
        jets_phi = flat_np_view(jets.phi[jets_mask])
        res = self.evaluator.evaluate("jetvetomap", jets_eta, jets_phi)

        # jet veto maps return 0 for a good jet
        res = ~ak.any(layout_ak_array(res, jets.pt[jets_mask]), axis=-1)

        return res
