import json
from pathlib import Path

import awkward
import numba
import numpy
import numpy as np
from numba import types
from numba.typed import Dict

from ...dataset import Dataset, DatasetType
from ...eras import Year

_numba_bool = None
if hasattr(types, "bool"):
    _numba_bool = types.bool
else:
    _numba_bool = types.bool_


def _make_lumi_mask_dict():
    return Dict.empty(key_type=types.uint32, value_type=types.uint32[:])


_lumi_mask_dict_type = numba.typeof(_make_lumi_mask_dict())


# From: https://github.com/scikit-hep/coffea/blob/master/src/coffea/lumi_tools/lumi_tools.py
class LumiMask:
    """
    Holds a luminosity mask index, and provides vectorized lookup, retaining only valid (run,lumisection) pairs.

    Parameters
    ----------
        jsonfile : str
            Path the the 'golden json' file or other valid lumiSection database in json format.

    This class parses a CMS lumi json into an efficient valid lumiSection lookup table.
    """

    def __init__(self, dataset: Dataset):
        self.dataset_type = dataset.dataset_type

        match dataset.year:
            case Year.RunSummer24:
                jsonfile = Path(
                    "/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions24/latest/Cert_Collisions2024_378981_386951_Golden.json"
                )
            case Year.RunSummer23BPix:
                raise NotImplementedError(dataset.year)
            case Year.RunSummer23:
                raise NotImplementedError(dataset.year)
            case Year.RunSummer22EE:
                raise NotImplementedError(dataset.year)
            case Year.RunSummer22:
                raise NotImplementedError(dataset.year)
            case Year.Run2018:
                raise NotImplementedError(dataset.year)
            case Year.Run2017:
                raise NotImplementedError(dataset.year)
            case Year.Run2016preVFP:
                raise NotImplementedError(dataset.year)
            case Year.Run2016postVFP:
                raise NotImplementedError(dataset.year)
            case _:
                raise ValueError(f"Invalid year {year}")

        with open(jsonfile) as fin:
            goldenjson = json.load(fin)

        self._masks = dict()

        for run, lumilist in goldenjson.items():
            mask = numpy.array(lumilist, dtype=numpy.uint32).flatten()
            mask[::2] -= 1
            self._masks[numpy.uint32(run)] = mask

    def __call__(self, runs, lumis):
        """
        Check pairs of runs and lumis for validity, and produce a mask retaining the valid pairs.

        Parameters
        ----------
            runs : numpy.ndarray or awkward.highlevel.Array
                Vectorized list of run numbers
            lumis : numpy.ndarray or awkward.highlevel.Array
                Vectorized list of lumiSection numbers

        Returns
        -------
            mask_out : numpy.ndarray
                An array of dtype `bool` where valid (run, lumi) tuples
                will have their corresponding entry set ``True``.
        """

        if self.dataset_type != DatasetType.DATA:
            return awkward.from_numpy(np.asarray(np.ones(len(runs)), dtype=np.bool))

        def apply(runs, lumis):
            # fill numba typed dict
            _masks = _make_lumi_mask_dict()
            for k, v in self._masks.items():
                _masks[k] = v

            runs_orig = runs
            if isinstance(runs, awkward.highlevel.Array):
                runs = awkward.to_numpy(
                    awkward.typetracer.length_zero_if_typetracer(runs)
                ).astype(numpy.uint32)
            if isinstance(lumis, awkward.highlevel.Array):
                lumis = awkward.to_numpy(
                    awkward.typetracer.length_zero_if_typetracer(lumis)
                ).astype(numpy.uint32)
            mask_out = numpy.zeros(dtype=bool, shape=runs.shape)
            LumiMask._apply_run_lumi_mask_kernel(_masks, runs, lumis, mask_out)
            if isinstance(runs_orig, awkward.Array):
                mask_out = awkward.Array(mask_out)
            if awkward.backend(runs_orig) == "typetracer":
                mask_out = awkward.Array(
                    mask_out.layout.to_typetracer(forget_length=True)
                )
            return mask_out

        return apply(runs, lumis)

    # This could be run in parallel, but windows does not support it
    @staticmethod
    @numba.njit(
        types.void(
            _lumi_mask_dict_type, types.uint32[:], types.uint32[:], _numba_bool[:]
        ),
        parallel=True,
        fastmath=True,
        cache=True,
    )
    def _apply_run_lumi_mask_kernel(masks, runs, lumis, mask_out):
        for iev in numba.prange(len(runs)):
            run = numpy.uint32(runs[iev])
            lumi = numpy.uint32(lumis[iev])

            if run in masks:
                lumimask = masks[run]
                ind = numpy.searchsorted(lumimask, lumi)
                if numpy.mod(ind, 2) == 1:
                    mask_out[iev] = 1


def _wrap_unique(array):
    out = numpy.unique(awkward.typetracer.length_one_if_typetracer(array), axis=0)

    if awkward.backend(array) == "typetracer":
        out = awkward.Array(
            out.layout.to_typetracer(forget_length=True),
            behavior=out.behavior,
            attrs=out.attrs,
        )
    return out
