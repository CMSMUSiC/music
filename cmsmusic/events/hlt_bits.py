import awkward as ak
import uproot


from .load_fields import Field, load_fields


def _build_hlt_bits(evts: uproot.TTree) -> ak.Array:
    HLT_BITS_PREFIX = "HLT_"

    _hlt_bits = load_fields(["HLT_IsoMu24"], evts)

    hlt_bits = ak.zip(
        {f: _hlt_bits[f] for f in ak.fields(_hlt_bits)},
    )
    return hlt_bits
