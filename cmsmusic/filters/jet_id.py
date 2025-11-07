import correctionlib
import awkward as ak
import numpy as np
from numpy.typing import NDArray
from enum import StrEnum

from cmsmusic.ak_utils import flat_np_view

from ..eras import Year
from ..ak_utils import *
from ..dataset import Dataset


class JetIdWP(StrEnum):
    AK4PUPPI_TightLeptonVeto = "AK4PUPPI_TightLeptonVeto"
    AK4PUPPI_Tight = "AK4PUPPI_Tight"


class JetId:
    def __init__(self, dataset: Dataset, jetid_wp: JetIdWP) -> None:
        self.year = dataset.year
        self.wp = jetid_wp

        match self.year:
            case Year.RunSummer24:
                self.evaluator = correctionlib.CorrectionSet.from_file(
                    "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/jetid.json.gz"
                )[self.wp]
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

    def __call__(self, jets: ak.Array) -> ak.Array:
        jets_eta = flat_np_view(jets.eta)  # type:ignore
        jets_chHEF = flat_np_view(jets.chHEF)  # type:ignore
        jets_neHEF = flat_np_view(jets.neHEF)  # type:ignore
        jets_chEmEF = flat_np_view(jets.chEmEF)  # type:ignore
        jets_neEmEF = flat_np_view(jets.neEmEF)  # type:ignore
        jets_muEF = flat_np_view(jets.muEF)  # type:ignore
        jets_chMultiplicity = flat_np_view(jets.chMultiplicity)  # type:ignore
        jets_neMultiplicity = flat_np_view(jets.neMultiplicity)  # type:ignore
        jets_multiplicity = jets_chMultiplicity + jets_neMultiplicity

        res = self.evaluator.evaluate(
            jets_eta,
            jets_chHEF,
            jets_neHEF,
            jets_chEmEF,
            jets_neEmEF,
            jets_muEF,
            jets_chMultiplicity,
            jets_neMultiplicity,
            jets_multiplicity,
        )

        res = layout_ak_array(res, jets.pt)  # type:ignore
        return res
