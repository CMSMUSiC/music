from enum import StrEnum


class Redirectors(StrEnum):
    FNAL = "root://cmsxrootd.fnal.gov//"
    INFN = "root://xrootd-cms.infn.it//"
    CERN = "root://cms-xrd-global.cern.ch//"
    RWTH = "root://grid-dcache.physik.rwth-aachen.de//"
