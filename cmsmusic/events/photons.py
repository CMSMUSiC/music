import awkward as ak
import uproot
import vector

vector.register_awkward()  # <- important


def _build_photons(evts: uproot.TTree) -> ak.Array:
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
