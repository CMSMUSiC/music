from numba import njit, types, int64, float64, boolean
from numba.experimental import jitclass
import numpy as np
import hist

# ---------- JITCLASS SPEC (added under/overflow fields) ----------
spec = [
    ("name", types.unicode_type),
    ("counts", types.float64[:]),  # length = nbins (no flow slots)
    ("variances", types.float64[:]),  # length = nbins
    (
        "edges",
        types.float64[:],
    ),  # length = nbins+1 for variable bins, or empty for uniform
    ("nbins", int64),
    ("is_uniform", boolean),
    ("low", float64),  # used for uniform bins
    ("high", float64),
    ("width", float64),
    # underflow / overflow scalars and their variances
    ("underflow", float64),
    ("overflow", float64),
    ("underflow_variance", float64),
    ("overflow_variance", float64),
]


# ---------- CLASS DEFINITION ----------
@jitclass(spec)  # type: ignore
class Hist:
    def __init__(
        self, name, counts, variances, edges, nbins, is_uniform, low, high, width
    ):
        self.name = name
        self.counts = counts
        self.variances = variances
        self.edges = edges
        self.nbins = nbins
        self.is_uniform = is_uniform
        self.low = low
        self.high = high
        self.width = width

        # initialize flow bins (scalars)
        self.underflow = 0.0
        self.overflow = 0.0
        self.underflow_variance = 0.0
        self.overflow_variance = 0.0

    def add_to_bin(self, idx, weight):
        self.counts[idx] += weight
        self.variances[idx] += weight * weight

    def _add_underflow(self, weight):
        self.underflow += weight
        self.underflow_variance += weight * weight

    def _add_overflow(self, weight):
        self.overflow += weight
        self.overflow_variance += weight * weight

    def fill(self, x, weight=1.0):
        """
        Fill a single value `x` with weight `weight`.
        Behavior:
          - values inside range -> increment corresponding bin
          - values below axis -> underflow scalar increment
          - values above axis -> overflow scalar increment
        Returns True if the value was placed somewhere (including flows).
        """
        # handle NaN/inf conservatively: consider them not filled
        if not np.isfinite(x):
            return False

        if self.is_uniform:
            # under / overflow check relative to [low, high)
            if x < self.low:
                self._add_underflow(weight)
                return True
            if x >= self.high:
                self._add_overflow(weight)
                return True

            # normal bin calculation (guaranteed 0 <= idx < nbins)
            idx = int((x - self.low) / self.width)
            # boundary check (just in case of floating rounding)
            if idx < 0:
                self._add_underflow(weight)
                return True
            if idx >= self.nbins:
                self._add_overflow(weight)
                return True

            self.counts[idx] += weight
            self.variances[idx] += weight * weight
            return True
        else:
            # variable-width: searchsorted -> idx in [-1 .. nbins]
            idx = np.searchsorted(self.edges, x, side="right") - 1
            if idx < 0:
                self._add_underflow(weight)
                return True
            if idx >= self.nbins:
                self._add_overflow(weight)
                return True

            self.counts[idx] += weight
            self.variances[idx] += weight * weight
            return True

    def clear(self):
        # clear inner bins
        for i in range(self.nbins):
            self.counts[i] = 0.0
            self.variances[i] = 0.0
        # clear flows
        self.underflow = 0.0
        self.overflow = 0.0
        self.underflow_variance = 0.0
        self.overflow_variance = 0.0


# ---------- FACTORY FUNCTIONS (compile-time) ----------
@njit
def make_uniform_hist(bins: int, low: float, high: float, name: str = "hist"):
    assert bins > 0
    counts = np.zeros(bins, dtype=np.float64)
    variances = np.zeros(bins, dtype=np.float64)
    edges = np.zeros(0, dtype=np.float64)
    width = (high - low) / bins
    # Hist __init__ will initialize flow scalars
    return Hist(name, counts, variances, edges, bins, True, low, high, width)


@njit
def make_variable_hist(edges_in, name: str = "hist"):
    # edges_in is a numpy 1D array of length nbins+1 (monotonic)
    nbins = len(edges_in) - 1
    counts = np.zeros(nbins, dtype=np.float64)
    variances = np.zeros(nbins, dtype=np.float64)
    # copy edges into a new array (to ensure typed ownership)
    edges = np.empty(len(edges_in), dtype=np.float64)
    for i in range(len(edges_in)):
        edges[i] = edges_in[i]
    return Hist(name, counts, variances, edges, nbins, False, 0.0, 0.0, 0.0)


# ---------- EXAMPLES: using the Hist inside njit ----------
@njit
def example_fill_uniform_with_flows():
    h = make_uniform_hist(4, 0.0, 1.0, "u")
    h.fill(-0.1, 2.0)  # underflow
    h.fill(0.05, 1.0)  # bin 0
    h.fill(0.95, 3.0)  # bin 3 (since 4 bins [0,0.25,0.5,0.75,1.0))
    h.fill(1.5, 4.0)  # overflow
    return h


@njit
def example_fill_variable_with_flows():
    e = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float64)  # 3 bins
    h = make_variable_hist(e, "v")
    h.fill(-1.0, 1.0)  # underflow
    h.fill(0.1, 2.0)  # bin 0
    h.fill(0.8, 3.0)  # bin 2
    h.fill(2.0, 4.0)  # overflow
    return h


def to_hist(jit_hist):
    """
    Convert a Numba jitclass `Hist` instance (with scalar underflow/overflow and their variances)
    into a scikit-hep `hist.Hist` object.

    Expects the jitclass layout from the last example:
      - jit_hist.nbins : number of inner bins
      - jit_hist.counts : length == nbins (inner bins only)
      - jit_hist.variances : length == nbins (inner bins only)
      - jit_hist.underflow, jit_hist.overflow : scalar under/overflow totals
      - jit_hist.underflow_variance, jit_hist.overflow_variance : scalar variances for flows
      - jit_hist.is_uniform (bool), jit_hist.low, jit_hist.high or jit_hist.edges

    Returns:
      hist.Hist
    """

    nbins = int(jit_hist.nbins)

    # inner bin arrays
    inner_counts = np.array(jit_hist.counts, dtype=float)
    inner_vars = np.array(jit_hist.variances, dtype=float)

    # flows (scalars). Use getattr with fallbacks to 0.0 if absent.
    under_val = float(getattr(jit_hist, "underflow", 0.0))
    over_val = float(getattr(jit_hist, "overflow", 0.0))
    under_var = float(getattr(jit_hist, "underflow_variance", 0.0))
    over_var = float(getattr(jit_hist, "overflow_variance", 0.0))

    # decide whether we need weighted storage (variances present)
    has_var = np.any(inner_vars != 0.0) or (under_var != 0.0) or (over_var != 0.0)

    # safe axis name handling: only pass a string name if present and non-empty
    axis_name = getattr(jit_hist, "name", None)
    name_kw = {}
    if axis_name is not None:
        # convert to str (handles numpy.str_, etc.), but only pass non-empty string
        s = str(axis_name)
        if s != "":
            name_kw = {"name": s}

    # build axis: ALWAYS enable underflow/overflow so we can set flows
    if bool(jit_hist.is_uniform):
        low = float(jit_hist.low)
        high = float(jit_hist.high)
        axis = hist.axis.Regular(
            nbins, low, high, underflow=True, overflow=True, **name_kw
        )
    else:
        edges = np.array(jit_hist.edges, dtype=float)
        axis = hist.axis.Variable(edges, underflow=True, overflow=True, **name_kw)

    # create histogram with appropriate storage and fill inner bins + flows
    if has_var:
        h = hist.Hist(axis, storage=hist.storage.Weight())
        # inner bins: provide (value, variance) pairs
        paired = np.empty((nbins, 2), dtype=float)
        paired[:, 0] = inner_counts
        paired[:, 1] = inner_vars
        h[0:nbins] = paired

        # assign flows as [value, variance] pairs
        h[hist.underflow] = [under_val, under_var]
        h[hist.overflow] = [over_val, over_var]
    else:
        h = hist.Hist(axis)  # default Double storage
        h[0:nbins] = inner_counts
        h[hist.underflow] = float(under_val)
        h[hist.overflow] = float(over_val)

    return h
