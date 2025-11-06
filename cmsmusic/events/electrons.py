import awkward as ak
import uproot
import vector

vector.register_awkward()  # <- important


from .load_fields import load_fields, Field

ELECTRON_MASS = 0.000511


def _build_electrons(evts: uproot.TTree) -> ak.Array:
    ELECTRON_PREFIX = "Electron_"

    _electrons = load_fields(
        [
            "Electron_pt",
            "Electron_eta",
            "Electron_phi",
            Field("Electron_mass", ELECTRON_MASS),
            "Electron_charge",
        ],
        evts,
    )

    electrons = ak.zip(
        {f[len(ELECTRON_PREFIX) :]: _electrons[f] for f in ak.fields(_electrons)},
        with_name="Momentum4D",
    )
    return electrons
