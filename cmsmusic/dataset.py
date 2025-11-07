import getpass
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import StrEnum
from typing import Self

import uproot
from dbs.apis.dbsClient import DbsApi
from pydantic import BaseModel, model_validator
from rich.progress import track

from .eras import LHCRun, NanoADODVersion, Year
from .redirectors import Redirectors

logger = logging.getLogger("Datasets")
dbs = DbsApi("https://cmsweb.cern.ch/dbs/prod/global/DBSReader")

try:
    os.environ["USER"]

except KeyError as _:
    os.environ["USER"] = getpass.getuser()


class ProcessGroup(StrEnum):
    DATA = "Data"
    DRELL_YAN = "Drell-Yan"
    QCD = "QCD"
    TTBAR = "ttbar"
    WJETS = "WJets"
    ZZ = "ZZ"
    WW = "WW"
    ZGAMMA = "ZGamma"
    WGAMMA = "WGamma"


class DatasetType(StrEnum):
    DATA = "Data"
    BACKGROUND = "Background"
    SIGNAL = "Signal"


def get_sum_weights(evts: uproot.TTree, dataset_type: DatasetType) -> tuple[float, int]:
    num_events = int(evts.num_entries)
    if dataset_type == DatasetType.DATA:
        return float(num_events), num_events

    import awkward as ak

    has_LHEWeight_originalXWGTUP = True
    LHEWeight_originalXWGTUP = None
    try:
        LHEWeight_originalXWGTUP = ak.sum(
            evts.arrays(["LHEWeight_originalXWGTUP"])["LHEWeight_originalXWGTUP"]
        )
    except:
        has_LHEWeight_originalXWGTUP = False

    has_genWeight = True
    all_genWeight_are_one = False
    genWeight = None
    try:
        genWeight = ak.sum(evts["genWeight"])
        all_genWeight_are_one = ak.all(evts.array("genWeight") == 1)
    except:
        has_genWeight = False

    if has_genWeight:
        if not all_genWeight_are_one:
            assert genWeight is not None
            return genWeight, num_events

    if has_LHEWeight_originalXWGTUP:
        assert LHEWeight_originalXWGTUP is not None
        return LHEWeight_originalXWGTUP, num_events

    raise RuntimeError("could not compute sum of genWeights")


def test_file(f: str, dataset_type: DatasetType) -> tuple[bool, str, float, int]:
    success = False
    sum_weights = None
    num_events = None
    for redirector in Redirectors:
        try:
            evts = uproot.open(f"{redirector}{f}:Events")
            sum_weights, num_events = get_sum_weights(evts, dataset_type)  # type:ignore
            success = True
            break
        except:
            continue

    assert sum_weights is not None
    assert num_events is not None
    return success, f, sum_weights, num_events


class Dataset(BaseModel):
    das_names: str | list[str]
    process_name: str | None = None
    process_group: ProcessGroup
    year: Year
    nanoadod_version: NanoADODVersion
    lhc_run: LHCRun
    dataset_type: DatasetType
    xsec: float
    filter_eff: float
    k_factor: float
    lfns: list[str] | None = None
    generator_filter: str | None = None
    sum_weights: float | None = None
    num_events: int | None = None

    def short_str(self) -> str:
        return f"[{self.process_name}]_[{self.process_group}]_[{self.year}]_[{self.lhc_run}]_[{self.dataset_type}]"

    @model_validator(mode="after")
    def set_xsec(self) -> Self:
        if self.dataset_type == DatasetType.DATA:
            self.xsec = 1.0
            self.k_factor = 1.0
            self.filter_eff = 1.0

        return self

    @model_validator(mode="after")
    def build_das_names(self) -> Self:
        if isinstance(self.das_names, str):
            self.das_names = [self.das_names]

        return self

    @model_validator(mode="after")
    def set_process_name(self) -> Self:
        if self.process_name is None:
            self.process_name = self.das_names[0].split("/")[1]

        if self.process_name is None or self.process_name == "":
            raise ValueError(f"Bad process name for {self}")

        return self

    @model_validator(mode="after")
    def build_lfn_list_and_sum_weights(self) -> Self:
        MIN_PERCENT_FILES = 0.6

        if self.lfns is None:
            self.lfns = []
            self.sum_weights = 0.0
            self.num_events = 0
            for das_name in self.das_names:
                logger.info(f"\nTesting files for {das_name}...")
                all_files = [
                    file["logical_file_name"].strip()
                    for file in dbs.listFiles(dataset=das_name)
                ]
                results = []
                sum_weights = 0.0
                num_events = 0
                with ProcessPoolExecutor() as ex:
                    futures = [
                        ex.submit(test_file, f, self.dataset_type) for f in all_files
                    ]
                    for fut in track(
                        as_completed(futures),
                        total=len(futures),
                        description=f"Processing...",
                    ):
                        success, f, _sum_weights, _num_events = fut.result()
                        if success:
                            results.append(f)
                            assert _sum_weights is not None
                            sum_weights += _sum_weights
                            assert _num_events is not None
                            num_events += _num_events

                if len(results) / len(all_files) < MIN_PERCENT_FILES:
                    raise RuntimeError(f"Not enough files passed test for {das_name}")

                self.lfns += results
                self.sum_weights += sum_weights
                self.num_events += num_events

        return self
