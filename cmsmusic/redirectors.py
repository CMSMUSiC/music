from enum import StrEnum


class Redirectors(StrEnum):
    FNAL = "root://cmsxrootd.fnal.gov//"
    INFN = "root://xrootd-cms.infn.it//"
    CERN = "root://cms-xrd-global.cern.ch//"
    RWTH = "root://grid-dcache.physik.rwth-aachen.de//"


def cycle_from(start: int | Redirectors):
    _redirectors = [r for r in Redirectors]

    match start:
        case int():
            pass
        case Redirectors():
            start = _redirectors.index(start)

    n = len(Redirectors)
    assert isinstance(start, int)
    for i in range(n):
        yield _redirectors[(start + i) % n]
