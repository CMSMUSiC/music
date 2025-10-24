import awkward as ak
import uproot
import vector

vector.register_awkward()  # <- important


def _build_electrons(evts: uproot.TTree) -> ak.Array:
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
