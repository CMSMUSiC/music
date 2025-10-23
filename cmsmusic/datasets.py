import getpass
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

dbs = DbsApi("https://cmsweb.cern.ch/dbs/prod/global/DBSReader")

try:
    os.environ["USER"]

except KeyError as _:
    os.environ["USER"] = getpass.getuser()


def test_file(f):
    success = False
    for redirector in Redirectors:
        try:
            uproot.open(f"{redirector}{f}")
            success = True
            break
        except:
            continue

    return success, f


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
    def build_lfn_list(self) -> Self:
        MIN_PERCENT_FILES = 0.6

        if self.lfns is None:
            self.lfns = []
            for das_name in self.das_names:
                print(f"Testing files for {das_name}...")
                all_files = [
                    file["logical_file_name"].strip()
                    for file in dbs.listFiles(dataset=das_name)
                ]
                results = []
                with ProcessPoolExecutor() as ex:
                    futures = [ex.submit(test_file, f) for f in all_files]
                    for fut in track(
                        as_completed(futures),
                        total=len(futures),
                        description=f"Processing...",
                    ):
                        success, f = fut.result()
                        if success:
                            results.append(f)

                if len(results) / len(all_files) < MIN_PERCENT_FILES:
                    raise RuntimeError(f"Not enough files passed test for {das_name}")

                self.lfns += results

        return self
