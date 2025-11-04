import correctionlib
import numpy as np
from numpy.typing import NDArray

from ...eras import Year


class JetVetoMaps:
    def __init__(self, year: Year) -> None:
        self.year = year

        match year:
            case Year.RunSummer24:
                self.evaluator = correctionlib.CorrectionSet.from_file(
                    "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/jetvetomaps.json.gz"
                )["Summer24Prompt24_RunBCDEFGHI_V1"]
            # case Year.RunSummer23BPix:
            #     pass
            # case Year.RunSummer23:
            #     pass
            # case Year.RunSummer22EE:
            #     pass
            # case Year.RunSummer22:
            #     pass
            # case Year.Run2018:
            #     pass
            # case Year.Run2017:
            #     pass
            # case Year.Run2016preVFP:
            #     pass
            # case Year.Run2016postVFP:
            #     pass
            case _:
                raise ValueError("Invalid year {year}")

    def __call__(
        self, eta: NDArray[np.float64], phi: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        res = self.evaluator.evaluate("jetvetomap", eta, phi)

        match res:
            case float():
                res = np.array([res], dtype=np.float64)
            case int():  # why?? lets make pyright happy...
                res = np.array([res], dtype=np.float64)
            case _:
                pass

        return res
