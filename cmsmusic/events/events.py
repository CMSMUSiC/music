import logging
import subprocess
from pathlib import Path

import awkward as ak
from numpy.typing import NDArray

import uproot
from pydantic import BaseModel, ConfigDict

from cmsmusic.events.jet_veto_maps import JetVetoMaps

from ..redirectors import Redirectors
from ..datasets import Dataset

from .electrons import _build_electrons
from .jets import _build_jets
from .met import _build_met
from .muons import _build_muons
from .photons import _build_photons
from .taus import _build_taus
from ..datasets import Dataset

logger = logging.getLogger("Events")


def load_file(file_lfn: str, enable_cache: bool) -> uproot.TTree:
    if enable_cache:
        local_path = Path(f"nanoaod_files_cache/{file_lfn.replace('/', '_')}")
        if local_path.exists():
            logger.info("File already cached...")
            nanoaod_file = uproot.open(f"{str(local_path)}:Events")
            return nanoaod_file  # type: ignore
        else:
            logger.info(f"Caching {file_lfn}...")
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
    data: ak.Array
    event_filters = dict[str, list[NDArray | ak.Array]]

    def add_event_filter(
        self,
        filter_name: str,
        filter_mask: NDArray | ak.Array,
    ) -> None:
        if filter_name in self.event_filters.keys():
            raise RuntimeError(f"{filter_name} already in event_filters")

        self.event_filters |= {filter_name: filter_mask}


class EventsBuilder:
    def __init__(self, dataset: Dataset, file_index: int, enable_cache: bool) -> None:
        assert dataset.lfns is not None
        self.input_file = dataset.lfns[file_index]
        self.enable_cache = enable_cache
        self.dataset = dataset

    def build(self) -> Events:
        if self.input_file is None:
            raise ValueError("input_file not set")
        if self.enable_cache is None:
            raise ValueError("input_file not set")

        evts = load_file(self.input_file, self.enable_cache)

        logger.info(type(evts))

        muons = _build_muons(evts)
        logger.info(muons.pt)
        logger.info(muons.muons_pt_up)
        logger.info(muons.muons_pt_down)
        logger.info(muons.mass)
        logger.info(muons.muons_mass_up)
        logger.info(muons.muons_mass_down)
        logger.info(type(muons))

        electrons = _build_electrons(evts)
        taus = _build_taus(evts)
        photons = _build_photons(evts)
        jets = _build_jets(evts)
        met = _build_met(evts, jets)

        data = ak.zip(
            {
                "muons": muons,
                "electrons": electrons,
                "taus": taus,
                "photons": photons,
                "jets": jets,
                "met": met,
            },
            depth_limit=1,  # zip at the event level only
        )
        print(data)
        print(data.muons[1])
        print(data.muons.px)

        events = Events(data=data)

        jet_veto_maps = JetVetoMaps(self.dataset.year)
        assert not isinstance(jet_veto_maps, float)
        events.add_event_filter(
            "jet_veto_maps", jet_veto_maps(events.jets.eta, events.jets.phi)
        )

        return events
