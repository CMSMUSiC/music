import awkward as ak
from ...eras import Year
from ...datasets import Dataset


def compute_met_filters(flags, dataset: Dataset) -> ak.Array:
    match dataset.year:
        case Year.RunSummer24:
            return ak.Array(
                (flags.goodVertices == 1)
                & (flags.globalSuperTightHalo2016Filter == 1)
                & (flags.EcalDeadCellTriggerPrimitiveFilter == 1)
                & (flags.BadPFMuonFilter == 1)
                & (flags.BadPFMuonDzFilter == 1)
                & (flags.hfNoisyHitsFilter == 1)
                & (flags.eeBadScFilter == 1)
                & (flags.ecalBadCalibFilter == 1)
            )
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
            raise ValueError(f"Invalid year {dataset.year}")
