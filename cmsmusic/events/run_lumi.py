import awkward as ak
import uproot


from .load_fields import load_fields


def _build_run_lumi(evts: uproot.TTree) -> tuple[ak.Array, ak.Array]:
    _run_lumi = load_fields(
        [
            "luminosityBlock",
            "run",
        ],
        evts,
    )

    return _run_lumi.run, _run_lumi.luminosityBlock
