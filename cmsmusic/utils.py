from typing import Any

from numba import njit
import vector
import awkward as ak
import numpy as np
from numpy.typing import NDArray


# From columnflow
# https://github.com/columnflow/columnflow
def layout_ak_array(data_array: NDArray | ak.Array, layout_array: ak.Array) -> ak.Array:
    """
    Takes a *data_array* and structures its contents into the same structure as *layout_array*, with
    up to one level of nesting. In particular, this function can be used to create new awkward
    arrays from existing numpy arrays and forcing a known, potentially ragged shape to it. Example:

     .. code-block:: python
     │
     │   a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
     │   b = ak.Array([[], [0, 0], [], [0, 0, 0]])
     │
     │   c = layout_ak_array(a, b)
     │   # <Array [[], [1.0, 2.0], [], [3.0, 4.0, 5.0]] type='4 * var * float32'>
    """
    return ak.unflatten(
        ak.flatten(data_array, axis=None),  # type: ignore
        ak.num(layout_array, axis=1),
        axis=0,
    )


# From columnflow
# https://github.com/columnflow/columnflow
def flat_np_view(ak_array: Any, axis: int | None = None) -> NDArray:
    """
    Takes an *ak_array* and returns a fully flattened numpy view. The flattening is applied along
    *axis*. See *ak.flatten* for more info.
    """
    return np.asarray(ak.flatten(ak_array, axis=axis))  # type: ignore


@njit(inline="always")
def make_vector(obj):
    return vector.obj(pt=obj.pt, eta=obj.eta, phi=obj.phi, mass=obj.mass)


vec = make_vector


@njit(inline="always")
def make_null_vector():
    return vector.obj(pt=0.0, eta=0.0, phi=0.0, mass=0.0)


null_vec = make_null_vector
