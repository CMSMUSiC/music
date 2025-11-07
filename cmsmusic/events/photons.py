import awkward as ak
import uproot
import vector

from .load_fields import Field, load_fields

vector.register_awkward()  # <- important


def _build_photons(evts: uproot.TTree) -> ak.Array:
    PHOTON_PREFIX = "Photon_"

    fields: list[Field] = [
        Field("Photon_pt"),
        Field("Photon_eta"),
        Field("Photon_phi"),
        Field("Photon_mass", 0.0, "Photon_pt"),
    ]

    _photons = load_fields(fields, evts)

    photons = ak.zip(
        {f[len(PHOTON_PREFIX) :]: _photons[f] for f in ak.fields(_photons)},
        with_name="Momentum4D",
    )

    return photons
