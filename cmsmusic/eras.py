from enum import StrEnum, auto


class NanoADODVersion(StrEnum):
    V9 = auto()
    V10 = auto()
    V11 = auto()
    V12 = auto()
    V13 = auto()
    V14 = auto()
    V15 = auto()


class LHCRun(StrEnum):
    Run2 = "Run2"
    Run3 = "Run3"


class Year(StrEnum):
    RunSummer24 = "RunSummer24"
    RunSummer23BPix = "RunSummer23BPix"
    RunSummer23 = "RunSummer23"
    RunSummer22EE = "RunSummer22EE"
    RunSummer22 = "RunSummer22"
    Run2018 = "Run2018"
    Run2017 = "Run2017"
    Run2016preVFP = "Run2016preVFP"
    Run2016postVFP = "Run2016postVFP"
