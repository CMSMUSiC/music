import logging
import subprocess
from pathlib import Path

import awkward as ak
import uproot
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


from ..datasets import Dataset
from ..redirectors import Redirectors
from .corrections.jet_veto_maps import JetVetoMaps
from .corrections.lumi_filter import LumiMask
from .corrections.met_filters import compute_met_filters
from .electrons import _build_electrons
from .jets import _build_jets
from .met import _build_met
from .flags import _build_flags
from .run_lumi import _build_run_lumi
from .hlt_bits import _build_hlt_bits
from .trigobjs import _build_trigobjs
from .muons import _build_muons
from .photons import _build_photons
from .taus import _build_taus

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
    event_filters: dict[str, NDArray | ak.Array] = {}

    @property
    def num_events(self) -> int:
        return len(self.data)

    def add_event_filter(
        self,
        filter_name: str,
        filter_mask: NDArray | ak.Array,
    ) -> None:
        if filter_name in self.event_filters.keys():
            raise ValueError(f"{filter_name} already in event_filters")

        self.event_filters |= {filter_name: filter_mask}

    def get_event_filter(self, *, block_list: list[str] = []):
        if len(self.event_filters) == 0:
            raise RuntimeError("No event filter has been set")

        _event_filter = ak.ones_like(
            self.event_filters[next(iter(self.event_filters.keys()))]
        )
        for filter_name in self.event_filters:
            if filter_name not in block_list:
                _event_filter = _event_filter & self.event_filters[filter_name]

        return _event_filter


class EventsBuilder:
    def __init__(self, dataset: Dataset, file_index: int, enable_cache: bool) -> None:
        assert dataset.lfns is not None
        self.input_file = dataset.lfns[file_index]
        self.enable_cache = enable_cache
        self.dataset = dataset

    def build(self) -> Events:
        evts = load_file(self.input_file, self.enable_cache)

        logger.info(type(evts))

        run, lumi = _build_run_lumi(evts)
        hlt_bits = _build_hlt_bits(evts)
        trigobjs = _build_trigobjs(evts)
        muons = _build_muons(evts)
        electrons = _build_electrons(evts)
        taus = _build_taus(evts)
        photons = _build_photons(evts)
        jets = _build_jets(evts, self.dataset)
        met = _build_met(evts, jets)
        flags = _build_flags(evts)

        data = ak.zip(
            {
                "run": run,
                "luminosityBlock": lumi,
                "hlt_bits": hlt_bits,
                "trigobjs": trigobjs,
                "muons": muons,
                "electrons": electrons,
                "taus": taus,
                "photons": photons,
                "jets": jets,
                "met": met,
                "flags": flags,
            },
            depth_limit=1,  # zip at the event level only
        )
        print(data)
        print(data.muons[1])
        print(data.muons.px)
        print(data.muons.pt)

        events = Events(data=ak.Array(data))

        lumi_mask = LumiMask(self.dataset)
        events.add_event_filter(
            "run_lumi_filter",
            lumi_mask(events.data.run, events.data.luminosityBlock),
        )

        events.add_event_filter(
            "met_filters",
            compute_met_filters(events.data.flags, self.dataset),
        )

        jet_veto_maps = JetVetoMaps(self.dataset)
        events.add_event_filter(
            "jet_veto_maps",
            jet_veto_maps(events.data.jets, events.data.muons),
        )
        print(events.event_filters)

        return events
