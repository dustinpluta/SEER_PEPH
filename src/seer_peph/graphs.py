"""
seer_peph/graphs.py
===================
Adjacency graph utilities for the BYM2 spatial frailty model.

Responsibilities
----------------
1. Represent an area-level adjacency graph in the sparse format required by
   the NumPyro ICAR implementation (edge list: node1, node2).
2. Compute the BYM2 scaling factor for a given graph so that the ICAR
   component has marginal variance 1 (Riebler et al. 2016).
3. Generate synthetic connected graphs for simulation studies:
      make_ring_lattice(A, k)  — ring with k neighbours per node
      make_grid(nrow, ncol)    — rectangular grid
4. Load real adjacency from a dense matrix or edge-list CSV
   (used for the Georgia county graph in Phase 7).
5. Validate graphs for connectivity and structural correctness.

BYM2 background
---------------
The BYM2 prior (Riebler et al. 2016) decomposes the area-level frailty as:

    u_a = sigma * (sqrt(phi) * s_a + sqrt(1 - phi) * epsilon_a)

where:
    s       ~ ICAR(graph)       spatially structured component (sum-to-zero)
    epsilon ~ N(0, I)           unstructured IID component
    sigma   > 0                 overall frailty SD
    phi     in (0, 1)           proportion of variance that is spatial

The scaling_factor is chosen so that Var(s_a) ≈ 1 marginally, making sigma
interpretable as the total SD regardless of graph topology. It is the
geometric mean of the diagonal of the Moore-Penrose pseudoinverse of
Q = D - W (the ICAR precision matrix).

ICAR log-density (sparse edge-list form)
-----------------------------------------
    log p(s) ∝ -1/2 * sum_{(a,b) in edges} (s_a - s_b)^2

Evaluated in NumPyro as:
    numpyro.factor("icar", -0.5 * jnp.sum((s[node1] - s[node2]) ** 2))

Sum-to-zero constraint
----------------------
The ICAR prior is improper without a constraint. Enforce it by
parameterising the spatial field with A-1 free components; the last
component is their negative sum. See SpatialGraph.free_to_full().

Public API
----------
    SpatialGraph              dataclass holding all graph artefacts
    make_ring_lattice(A, k)   synthetic ring-lattice graph
    make_grid(nrow, ncol)     synthetic rectangular grid graph
    from_adjacency(W, name)   build SpatialGraph from dense matrix
    from_edge_csv(path, A)    build SpatialGraph from edge-list CSV
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import linalg


# ── SpatialGraph dataclass ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpatialGraph:
    """
    All graph artefacts needed by the NumPyro spatial frailty model.

    Attributes
    ----------
    A : int
        Number of areas (nodes).
    n_edges : int
        Number of undirected edges.
    node1 : np.ndarray, shape (n_edges,), dtype int32
        First node of each edge; node1[e] < node2[e] for all e
        (upper-triangle convention).
    node2 : np.ndarray, shape (n_edges,), dtype int32
        Second node of each edge.
    scaling_factor : float
        BYM2 scaling factor: geometric mean of diag(Q^+). Used inside
        the NumPyro model to scale the ICAR component to unit marginal
        variance before multiplying by sigma.
    adjacency : np.ndarray, shape (A, A), dtype float64
        Dense symmetric adjacency matrix (0/1 entries). Stored for
        diagnostics and computing derived quantities; not passed to NumPyro.
    name : str
        Human-readable label (e.g. "ring_lattice_A20_k4").

    Notes
    -----
    node1 and node2 use 0-based integer indices into the length-A frailty
    vector. They are stored as int32 to match JAX's default integer dtype.
    """

    A: int
    n_edges: int
    node1: np.ndarray
    node2: np.ndarray
    scaling_factor: float
    adjacency: np.ndarray
    name: str

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def degrees(self) -> np.ndarray:
        """Degree (number of neighbours) for each node."""
        return self.adjacency.sum(axis=1).astype(int)

    @property
    def mean_degree(self) -> float:
        """Mean degree across all nodes."""
        return float(self.degrees.mean())

    @property
    def is_connected(self) -> bool:
        """True iff the graph has exactly one connected component."""
        return _is_connected(self.adjacency)

    # ── ICAR constraint helper ────────────────────────────────────────────────

    def free_to_full(self, s_free: np.ndarray) -> np.ndarray:
        """
        Expand A-1 free ICAR parameters to a length-A sum-to-zero vector.

        The last component is set to -sum(s_free), enforcing sum(s) = 0
        exactly. Use this inside the NumPyro model to parameterise the
        ICAR field:

            s_free = numpyro.sample("s_free",
                         dist.Normal(0, 1).expand([graph.A - 1]))
            s_full = graph.free_to_full(s_free)

        Works with both NumPy and JAX arrays. JAX is used automatically
        when available and s_free is a JAX array.

        Parameters
        ----------
        s_free : array-like, shape (..., A-1)
            Free ICAR components.

        Returns
        -------
        array, shape (..., A)
            Sum-to-zero spatial field.
        """
        # Use JAX operations when JAX arrays are passed, otherwise NumPy.
        try:
            import jax.numpy as jnp
            if hasattr(s_free, "device"):   # JAX array
                last = -jnp.sum(s_free, axis=-1, keepdims=True)
                return jnp.concatenate([s_free, last], axis=-1)
        except ImportError:
            pass
        s_free = np.asarray(s_free)
        last = -s_free.sum(axis=-1, keepdims=True)
        return np.concatenate([s_free, last], axis=-1)

    # ── ICAR log-density (for testing / validation) ───────────────────────────

    def icar_logdens(self, s: np.ndarray) -> float:
        """
        Evaluate the unnormalised ICAR log-density for a given field s.

        Uses the sparse edge-list formula:
            log p(s) ∝ -1/2 * sum_{edges} (s[node1] - s[node2])^2

        Intended for testing and validation only. Inside NumPyro use
        numpyro.factor() directly with JAX arrays.

        Parameters
        ----------
        s : np.ndarray, shape (A,)
            Spatial field. Should satisfy sum(s) ≈ 0.

        Returns
        -------
        float
            Unnormalised log-density value.
        """
        s = np.asarray(s, dtype=float)
        if not np.isclose(s.sum(), 0.0, atol=1e-8):
            warnings.warn(
                f"ICAR field does not sum to zero (sum={s.sum():.4e}). "
                "Use free_to_full() to enforce the sum-to-zero constraint."
            )
        return float(-0.5 * np.sum((s[self.node1] - s[self.node2]) ** 2))

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the graph."""
        deg = self.degrees
        lines = [
            f"SpatialGraph : {self.name}",
            f"  Areas      : {self.A}",
            f"  Edges      : {self.n_edges}",
            f"  Connected  : {self.is_connected}",
            f"  Degrees    : min={deg.min()}  mean={deg.mean():.2f}  max={deg.max()}",
            f"  BYM2 scale : {self.scaling_factor:.6f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SpatialGraph(name={self.name!r}, A={self.A}, "
            f"n_edges={self.n_edges}, "
            f"scaling_factor={self.scaling_factor:.4f})"
        )


# ── internal helpers ───────────────────────────────────────────────────────────

def _validate_adjacency(W: np.ndarray, name: str = "") -> None:
    """
    Assert that W is a valid undirected binary adjacency matrix.

    Checks
    ------
    - Square 2-D array
    - Symmetric
    - Binary (0/1 entries only)
    - No self-loops (diagonal all zero)
    - At least one edge
    """
    label = f" ({name})" if name else ""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(
            f"Adjacency matrix{label} must be square 2-D, got shape {W.shape}"
        )
    if not np.allclose(W, W.T, atol=1e-12):
        raise ValueError(f"Adjacency matrix{label} must be symmetric")
    if not np.all((W == 0) | (W == 1)):
        raise ValueError(
            f"Adjacency matrix{label} must be binary (0/1 entries only)"
        )
    if np.any(np.diag(W) != 0):
        raise ValueError(
            f"Adjacency matrix{label} must have zero diagonal (no self-loops)"
        )
    if W.sum() == 0:
        raise ValueError(f"Adjacency matrix{label} has no edges")


def _is_connected(W: np.ndarray) -> bool:
    """
    Return True iff the graph represented by W is connected.

    Uses eigenvalue analysis: a connected graph's ICAR precision matrix
    Q = D - W has exactly one zero eigenvalue (the algebraic connectivity
    / Fiedler value lambda_2 > 0).
    """
    D = np.diag(W.sum(axis=1))
    Q = D - W
    evals = np.sort(np.linalg.eigvalsh(Q))
    return bool(
        np.isclose(evals[0], 0.0, atol=1e-10) and evals[1] > 1e-10
    )


def _edge_list(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the upper-triangle edge list from an adjacency matrix.

    Returns
    -------
    node1, node2 : np.ndarray, dtype int32
        Length-n_edges arrays with node1[e] < node2[e] for all e.
    """
    rows, cols = np.where(np.triu(W, k=1) > 0)
    return rows.astype(np.int32), cols.astype(np.int32)


def _bym2_scaling_factor(W: np.ndarray) -> float:
    """
    Compute the BYM2 scaling factor for a given adjacency matrix.

    Definition (Riebler et al. 2016)
    ---------------------------------
    Let Q = D - W be the ICAR precision matrix (rank A-1).
    Let Q^+ be its Moore-Penrose pseudoinverse.
    The scaling factor is:

        s = exp( mean( log( diag(Q^+) ) ) )
          = geometric mean of the marginal variances of the ICAR field

    Dividing the raw ICAR vector by sqrt(s) — or equivalently passing s
    to the model and scaling there — ensures the ICAR component has unit
    marginal variance, making sigma in the BYM2 decomposition interpretable
    as the total area-level frailty SD.

    Parameters
    ----------
    W : np.ndarray, shape (A, A)
        Pre-validated binary symmetric adjacency matrix.

    Returns
    -------
    float
        Positive scaling factor.
    """
    D = np.diag(W.sum(axis=1))
    Q = D - W
    Q_pinv = linalg.pinv(Q)
    diag_vals = np.diag(Q_pinv)
    if np.any(diag_vals <= 0):
        raise RuntimeError(
            "Non-positive diagonal entries in pseudoinverse of Q. "
            "Verify that the graph is connected and the adjacency matrix "
            "is correctly specified."
        )
    return float(np.exp(np.mean(np.log(diag_vals))))


def _build(W: np.ndarray, name: str) -> SpatialGraph:
    """
    Build a SpatialGraph from a pre-validated adjacency matrix.

    Internal factory used by all public constructors.
    """
    _validate_adjacency(W, name=name)
    if not _is_connected(W):
        raise ValueError(
            f"Graph '{name}' is not connected. "
            "All areas must belong to a single connected component for "
            "the ICAR prior to be well-defined."
        )
    node1, node2 = _edge_list(W)
    scaling_factor = _bym2_scaling_factor(W)
    return SpatialGraph(
        A=int(W.shape[0]),
        n_edges=int(len(node1)),
        node1=node1,
        node2=node2,
        scaling_factor=scaling_factor,
        adjacency=W.copy(),
        name=name,
    )


# ── synthetic graph constructors ───────────────────────────────────────────────

def make_ring_lattice(A: int, k: int = 4) -> SpatialGraph:
    """
    Build a k-regular ring lattice with A nodes.

    Each node i is connected to its k//2 nearest neighbours on each side
    around the ring (modular arithmetic). This gives a connected, regular
    graph where every node has exactly k neighbours.

    Ring lattices are the default synthetic graph for simulation studies
    (Option B) because they are deterministic, parameterised by a single
    integer k, connected by construction, and scale easily to any A.

    Parameters
    ----------
    A : int
        Number of nodes (areas). Must be >= k + 1.
    k : int
        Number of neighbours per node. Must be even and satisfy k >= 2.
        Default 4 (each node connected to 2 neighbours on each side).

    Returns
    -------
    SpatialGraph
        Named "ring_lattice_A{A}_k{k}".

    Examples
    --------
    Simulation graphs used across phases:
        make_ring_lattice(20,  k=4)   # Phase 1-5 development
        make_ring_lattice(50,  k=4)   # Phase 6 scaling study
        make_ring_lattice(159, k=4)   # Phase 6 scaling study (GA proxy)
    """
    if k < 2 or k % 2 != 0:
        raise ValueError(f"k must be a positive even integer, got k={k}")
    if A < k + 1:
        raise ValueError(f"A must be >= k+1, got A={A}, k={k}")

    W = np.zeros((A, A), dtype=float)
    half = k // 2
    for i in range(A):
        for d in range(1, half + 1):
            j = (i + d) % A
            W[i, j] = 1.0
            W[j, i] = 1.0

    return _build(W, name=f"ring_lattice_A{A}_k{k}")


def make_grid(nrow: int, ncol: int) -> SpatialGraph:
    """
    Build a rectangular grid graph with nrow × ncol nodes.

    Each interior node has 4 neighbours (N/S/E/W); edge and corner nodes
    have 3 and 2 neighbours respectively. Node indices run row-major:
    node (r, c) → index r * ncol + c.

    Useful as an alternative synthetic graph when a spatially embedded
    (non-regular) structure is desired.

    Parameters
    ----------
    nrow, ncol : int
        Grid dimensions. Must satisfy nrow >= 2 and ncol >= 2.

    Returns
    -------
    SpatialGraph
        Named "grid_{nrow}x{ncol}".
    """
    if nrow < 2 or ncol < 2:
        raise ValueError(f"Grid dimensions must be >= 2x2, got {nrow}x{ncol}")

    A = nrow * ncol
    W = np.zeros((A, A), dtype=float)
    for r in range(nrow):
        for c in range(ncol):
            i = r * ncol + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrow and 0 <= cc < ncol:
                    j = rr * ncol + cc
                    W[i, j] = 1.0

    return _build(W, name=f"grid_{nrow}x{ncol}")


# ── real-data constructors ─────────────────────────────────────────────────────

def from_adjacency(W: np.ndarray, name: str = "custom") -> SpatialGraph:
    """
    Build a SpatialGraph from a dense adjacency matrix.

    Parameters
    ----------
    W : array-like, shape (A, A)
        Binary symmetric adjacency matrix with zero diagonal. Will be
        cast to float64.
    name : str
        Label for the graph. Default "custom".

    Returns
    -------
    SpatialGraph

    Raises
    ------
    ValueError
        If W fails any structural validation check or is not connected.
    """
    W = np.asarray(W, dtype=float)
    return _build(W, name=name)


def from_edge_csv(
    path: str | Path,
    A: int,
    *,
    node1_col: str = "node1",
    node2_col: str = "node2",
    name: Optional[str] = None,
) -> SpatialGraph:
    """
    Build a SpatialGraph from an edge-list CSV file.

    The CSV must contain two columns of 0-based integer node indices.
    Each edge should appear exactly once (either direction is accepted;
    duplicates are silently deduplicated).

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    A : int
        Total number of nodes (areas). Required because isolated nodes
        would not appear in the edge list.
    node1_col, node2_col : str
        Column names for the two endpoint indices.
    name : str | None
        Label for the graph. Defaults to the CSV filename stem.

    Returns
    -------
    SpatialGraph

    Raises
    ------
    ValueError
        If the CSV is malformed or the resulting graph is invalid.

    CSV format example
    ------------------
    node1,node2
    0,1
    0,4
    1,2
    ...
    """
    import pandas as pd

    path = Path(path)
    if name is None:
        name = path.stem

    df = pd.read_csv(path)
    if node1_col not in df.columns or node2_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{node1_col}' and '{node2_col}'. "
            f"Found: {list(df.columns)}"
        )

    n1 = df[node1_col].to_numpy(dtype=int)
    n2 = df[node2_col].to_numpy(dtype=int)

    if n1.min() < 0 or n2.min() < 0:
        raise ValueError("Node indices must be non-negative (0-based)")
    if n1.max() >= A or n2.max() >= A:
        raise ValueError(
            f"Node indices exceed A-1={A-1}. "
            "Check that A is correct and indices are 0-based."
        )

    W = np.zeros((A, A), dtype=float)
    for a, b in zip(n1, n2):
        if a == b:
            raise ValueError(f"Self-loop detected at node {a}")
        W[a, b] = 1.0
        W[b, a] = 1.0

    return _build(W, name=name)


def load_georgia_counties(path: Optional[str | Path] = None) -> SpatialGraph:
    """
    Load the Georgia 159-county adjacency graph.

    Phase 7 placeholder. Two loading strategies are supported:

    (a) CSV edge list (preferred for reproducibility):
        A pre-built edge list derived from the US Census TIGER/Line county
        shapefiles. Construct with:
            from_edge_csv("georgia_counties_adj.csv", A=159,
                          name="georgia_159_counties")

    (b) Shapefile (requires geopandas + libpysal):
        Compute queen contiguity directly from the shapefile. Example:
            import geopandas as gpd
            from libpysal.weights import Queen
            gdf = gpd.read_file("tl_2020_13_county/tl_2020_13_county.shp")
            gdf = gdf.sort_values("GEOID").reset_index(drop=True)
            w = Queen.from_dataframe(gdf)
            W = np.array(w.full()[0])
            graph = from_adjacency(W, name="georgia_159_counties")

    Parameters
    ----------
    path : str | Path | None
        Path to a pre-built edge-list CSV. If None, raises NotImplementedError
        with instructions.

    Returns
    -------
    SpatialGraph with A=159.
    """
    if path is None:
        raise NotImplementedError(
            "Georgia county adjacency graph not yet built. "
            "Options:\n"
            "  (a) Provide a pre-built edge-list CSV:\n"
            "      load_georgia_counties('georgia_counties_adj.csv')\n"
            "  (b) Build from Census shapefile using geopandas + libpysal:\n"
            "      see docstring for code snippet."
        )
    return from_edge_csv(Path(path), A=159, name="georgia_159_counties")
