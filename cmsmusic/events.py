from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias


import awkward as ak
import uproot
import vector
from pydantic import BaseModel, ConfigDict


from .redirectors import Redirectors

vector.register_awkward()  # <- important

ObjectArray: TypeAlias = vector.backends.awkward.MomentumArray4D


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


def _add_muon_corr(muons: ObjectArray) -> ObjectArray:
    SHIFT = 0.15

    muons = ak.with_field(muons, (muons * (1.0 + SHIFT)).pt, "muons_pt_up")
    muons = ak.with_field(muons, (muons * (1.0 + SHIFT)).mass, "muons_mass_up")

    muons = ak.with_field(muons, (muons * (1.0 - SHIFT)).pt, "muons_pt_down")
    muons = ak.with_field(muons, (muons * (1.0 - SHIFT)).mass, "muons_mass_down")

    return muons


def _build_muons(evts: uproot.TTree) -> ObjectArray:
    MUON_PREFIX = "Muon_"

    _muons = evts.arrays(
        [
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
    )

    if "Muon_mass" not in ak.fields(_muons):
        MUON_MASS = 0.105_658_374_5
        _muons = ak.with_field(
            _muons, ak.ones_like(_muons["Muon_pt"]) * MUON_MASS, "Muon_mass"
        )

    muons = ak.zip(
        {f[len(MUON_PREFIX) :]: _muons[f] for f in ak.fields(_muons)},
        with_name="Momentum4D",
    )

    muons = _add_muon_corr(muons)

    return muons


def _build_electrons(evts: uproot.TTree) -> ObjectArray:
    ELECTRON_PREFIX = "Electron_"

    _electrons = evts.arrays(
        [
            "Electron_pt",
            "Electron_eta",
            "Electron_phi",
            "Electron_mass",
            "Electron_charge",
        ]
    )

    if "Electron_mass" not in ak.fields(_electrons):
        ELECTRON_MASS = 0.000511
        _electrons = ak.with_field(
            _electrons,
            ak.ones_like(_electrons["Electron_pt"]) * ELECTRON_MASS,
            "Electron_mass",
        )

    electrons = ak.zip(
        {f[len(ELECTRON_PREFIX) :]: _electrons[f] for f in ak.fields(_electrons)},
        with_name="Momentum4D",
    )
    return electrons


def _build_taus(evts: uproot.TTree) -> ObjectArray:
    TAU_PREFIX = "Tau_"

    _taus = evts.arrays(
        [
            "Tau_pt",
            "Tau_eta",
            "Tau_phi",
            "Tau_mass",
            "Tau_charge",
        ]
    )

    if "Tau_mass" not in ak.fields(_taus):
        TAU_MASS = 1.7769
        _taus = ak.with_field(
            _taus,
            ak.ones_like(_taus["Tau_pt"]) * TAU_MASS,
            "Tau_mass",
        )

    taus = ak.zip(
        {f[len(TAU_PREFIX) :]: _taus[f] for f in ak.fields(_taus)},
        with_name="Momentum4D",
    )

    return taus


def _build_photons(evts: uproot.TTree) -> ObjectArray:
    PHOTON_PREFIX = "Photon_"

    _photons = evts.arrays(
        [
            "Photon_pt",
            "Photon_eta",
            "Photon_phi",
        ]
    )

    if "Photon_mass" not in ak.fields(_photons):
        _photons = ak.with_field(
            _photons,
            ak.zeros_like(_photons["Photon_pt"]),
            "Photon_mass",
        )

    photons = ak.zip(
        {f[len(PHOTON_PREFIX) :]: _photons[f] for f in ak.fields(_photons)},
        with_name="Momentum4D",
    )

    return photons


def _build_jets(evts: uproot.TTree) -> ObjectArray:
    JET_PREFIX = "Jet_"

    _jets = evts.arrays(
        [
            "Jet_pt",
            "Jet_eta",
            "Jet_phi",
            "Jet_mass",
        ]
    )

    jets = ak.zip(
        {f[len(JET_PREFIX) :]: _jets[f] for f in ak.fields(_jets)},
        with_name="Momentum4D",
    )
    return jets


def _build_met(evts: uproot.TTree, jets: ObjectArray) -> ObjectArray:
    MET_PREFIX = "PuppiMET_"

    _met = evts.arrays(
        [
            "PuppiMET_pt",
            "PuppiMET_phi",
            "PuppiMET_phiUnclusteredDown",
            "PuppiMET_phiUnclusteredUp",
            "PuppiMET_ptUnclusteredDown",
            "PuppiMET_ptUnclusteredUp",
        ]
    )

    if "PuppiMET_mass" not in ak.fields(_met):
        _met = ak.with_field(
            _met,
            ak.zeros_like(_met["PuppiMET_pt"]),
            "PuppiMET_mass",
        )

    if "PuppiMET_eta" not in ak.fields(_met):
        _met = ak.with_field(
            _met,
            ak.zeros_like(_met["PuppiMET_pt"]),
            "PuppiMET_eta",
        )

    met = ak.zip(
        {f[len(MET_PREFIX) :]: _met[f] for f in ak.fields(_met)},
        with_name="Momentum4D",
    )
    return met


class Events(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    muons: ObjectArray
    electrons: ObjectArray
    taus: ObjectArray
    photons: ObjectArray
    jets: ObjectArray
    met: ObjectArray

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
