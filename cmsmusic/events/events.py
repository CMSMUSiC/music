import subprocess
from pathlib import Path

import awkward as ak
import uproot
from pydantic import BaseModel, ConfigDict

from ..redirectors import Redirectors
from .electrons import _build_electrons
from .jets import _build_jets
from .met import _build_met
from .muons import _build_muons
from .photons import _build_photons
from .taus import _build_taus


def load_file(file_lfn: str, enable_cache: bool) -> uproot.TTree:
    if enable_cache:
        local_path = Path(f"nanoaod_files_cache/{file_lfn.replace('/', '_')}")
        if local_path.exists():
            print("File already cached...")
            nanoaod_file = uproot.open(f"{str(local_path)}:Events")
            return nanoaod_file  # type: ignore
        else:
            print(f"Caching {file_lfn}...")
            for redirector in Redirectors:
                try:
                    subprocess.run(
                        f"xrdcp {redirector}{file_lfn} {str(local_path)}",
                        check=True,
                        shell=True,
                    )
                    nanoaod_file = uproot.open(f"{str(local_path)}:Events")
                    return nanoaod_file  # type: ignore
                except:
                    continue
            raise RuntimeError("File is not accessible by any redirector")

    for redirector in Redirectors:
        try:
            nanoaod_file = uproot.open(f"{redirector}{file_lfn}:Events")
            return nanoaod_file  # type: ignore
        except:
            continue

    raise RuntimeError("File is not accessible by any redirector")


class Events(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    muons: ak.Array
    electrons: ak.Array
    taus: ak.Array
    photons: ak.Array
    jets: ak.Array
    met: ak.Array

    @staticmethod
    def build_events(input_file: str, enable_cache: bool) -> "Events":
        evts = load_file(input_file, enable_cache)
        print(type(evts))

        muons = _build_muons(evts)
        print(muons.pt)
        print(muons.muons_pt_up)
        print(muons.muons_pt_down)
        print(muons.mass)
        print(muons.muons_mass_up)
        print(muons.muons_mass_down)
        print(type(muons))

        electrons = _build_electrons(evts)
        taus = _build_taus(evts)
        photons = _build_photons(evts)
        jets = _build_jets(evts)
        met = _build_met(evts, jets)

        return Events(
            muons=muons,
            electrons=electrons,
            taus=taus,
            photons=photons,
            jets=jets,
            met=met,
        )
