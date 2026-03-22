# tests/test_graphs.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seer_peph.graphs import (
    from_adjacency,
    from_edge_csv,
    load_georgia_counties,
    make_grid,
    make_ring_lattice,
)


def test_make_ring_lattice_basic_structure():
    graph = make_ring_lattice(A=8, k=4)

    assert graph.A == 8
    assert graph.n_edges == 16  # A * k / 2
    assert graph.name == "ring_lattice_A8_k4"
    assert graph.is_connected is True

    # Regular degree-k graph
    assert np.array_equal(graph.degrees, np.full(8, 4))
    assert graph.mean_degree == pytest.approx(4.0)

    # Symmetric adjacency, zero diagonal
    assert graph.adjacency.shape == (8, 8)
    assert np.allclose(graph.adjacency, graph.adjacency.T)
    assert np.all(np.diag(graph.adjacency) == 0)

    # Edge list shape and ordering convention
    assert graph.node1.shape == (graph.n_edges,)
    assert graph.node2.shape == (graph.n_edges,)
    assert graph.node1.dtype == np.int32
    assert graph.node2.dtype == np.int32
    assert np.all(graph.node1 < graph.node2)

    # BYM2 scale should be positive
    assert graph.scaling_factor > 0


def test_make_ring_lattice_invalid_arguments():
    with pytest.raises(ValueError, match="positive even integer"):
        make_ring_lattice(A=8, k=3)

    with pytest.raises(ValueError, match="positive even integer"):
        make_ring_lattice(A=8, k=0)

    with pytest.raises(ValueError, match="A must be >= k\\+1"):
        make_ring_lattice(A=4, k=4)


def test_make_grid_basic_structure():
    graph = make_grid(nrow=3, ncol=4)

    assert graph.A == 12
    assert graph.name == "grid_3x4"
    assert graph.is_connected is True

    # 3x4 grid has 17 undirected edges:
    # horizontal = 3*(4-1)=9, vertical = (3-1)*4=8
    assert graph.n_edges == 17

    # Degree pattern for 3x4 grid:
    # corners: 4 nodes with degree 2
    # non-corner boundary: 6 nodes with degree 3
    # interior: 2 nodes with degree 4
    deg = np.sort(graph.degrees)
    assert deg.tolist() == [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4]
    assert graph.scaling_factor > 0


def test_make_grid_invalid_arguments():
    with pytest.raises(ValueError, match="must be >= 2x2"):
        make_grid(1, 4)

    with pytest.raises(ValueError, match="must be >= 2x2"):
        make_grid(4, 1)


def test_from_adjacency_builds_expected_graph():
    W = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=float,
    )

    graph = from_adjacency(W, name="cycle4")

    assert graph.A == 4
    assert graph.n_edges == 4
    assert graph.name == "cycle4"
    assert graph.is_connected is True
    assert np.array_equal(graph.degrees, np.array([2, 2, 2, 2]))
    assert graph.scaling_factor > 0


def test_from_adjacency_rejects_invalid_inputs():
    # Non-square
    with pytest.raises(ValueError, match="square 2-D"):
        from_adjacency(np.ones((3, 2)), name="bad")

    # Non-symmetric
    with pytest.raises(ValueError, match="must be symmetric"):
        from_adjacency(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                ],
                dtype=float,
            ),
            name="bad",
        )

    # Non-binary
    with pytest.raises(ValueError, match="must be binary"):
        from_adjacency(
            np.array(
                [
                    [0, 2],
                    [2, 0],
                ],
                dtype=float,
            ),
            name="bad",
        )

    # Self-loop
    with pytest.raises(ValueError, match="zero diagonal"):
        from_adjacency(
            np.array(
                [
                    [1, 0],
                    [0, 0],
                ],
                dtype=float,
            ),
            name="bad",
        )

    # No edges
    with pytest.raises(ValueError, match="has no edges"):
        from_adjacency(np.zeros((3, 3)), name="bad")

    # Disconnected
    with pytest.raises(ValueError, match="not connected"):
        from_adjacency(
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=float,
            ),
            name="bad",
        )


def test_free_to_full_enforces_sum_to_zero():
    graph = make_ring_lattice(A=5, k=2)
    s_free = np.array([0.2, -1.3, 0.7, 0.4])

    s_full = graph.free_to_full(s_free)

    assert s_full.shape == (5,)
    assert np.allclose(s_full[:-1], s_free, atol=1e-6)
    assert np.isclose(s_full.sum(), 0.0, atol=1e-6)
    assert np.isclose(s_full[-1], -s_free.sum(), atol=1e-6)


def test_icar_logdens_matches_manual_edge_sum():
    graph = make_ring_lattice(A=6, k=2)  # simple cycle
    s = graph.free_to_full(np.array([1.0, -0.5, 0.25, -0.75, 0.0]))

    expected = -0.5 * np.sum((s[graph.node1] - s[graph.node2]) ** 2)
    got = graph.icar_logdens(s)

    assert got == pytest.approx(expected)


def test_icar_logdens_warns_when_sum_to_zero_violated():
    graph = make_ring_lattice(A=5, k=2)
    s = np.array([1.0, 0.0, -0.5, 0.25, 0.1])  # does not sum to 0

    with pytest.warns(UserWarning, match="does not sum to zero"):
        val = graph.icar_logdens(s)

    assert isinstance(val, float)


def test_summary_and_repr_contain_key_fields():
    graph = make_grid(2, 3)

    text = graph.summary()
    rep = repr(graph)

    assert "SpatialGraph :" in text
    assert "Areas" in text
    assert "Edges" in text
    assert "Connected" in text
    assert "BYM2 scale" in text

    assert "SpatialGraph(" in rep
    assert "name=" in rep
    assert "A=6" in rep
    assert "n_edges=" in rep


def test_from_edge_csv_builds_graph_and_deduplicates_reverse_edges(tmp_path):
    # Reverse duplicates should be harmless because the adjacency matrix is binary.
    df = pd.DataFrame(
        {
            "node1": [0, 1, 2, 3, 1],
            "node2": [1, 2, 3, 0, 0],  # edges: (0,1), (1,2), (2,3), (3,0), reverse of (0,1)
        }
    )
    path = tmp_path / "edges.csv"
    df.to_csv(path, index=False)

    graph = from_edge_csv(path, A=4, name="csv_cycle4")

    assert graph.A == 4
    assert graph.n_edges == 4
    assert graph.name == "csv_cycle4"
    assert graph.is_connected is True
    assert np.array_equal(graph.degrees, np.array([2, 2, 2, 2]))


def test_from_edge_csv_validates_columns_and_indices(tmp_path):
    bad_cols = pd.DataFrame({"a": [0], "b": [1]})
    bad_cols_path = tmp_path / "bad_cols.csv"
    bad_cols.to_csv(bad_cols_path, index=False)

    with pytest.raises(ValueError, match="must contain columns"):
        from_edge_csv(bad_cols_path, A=2)

    neg_idx = pd.DataFrame({"node1": [-1], "node2": [0]})
    neg_idx_path = tmp_path / "neg_idx.csv"
    neg_idx.to_csv(neg_idx_path, index=False)

    with pytest.raises(ValueError, match="non-negative"):
        from_edge_csv(neg_idx_path, A=2)

    too_large = pd.DataFrame({"node1": [0], "node2": [2]})
    too_large_path = tmp_path / "too_large.csv"
    too_large.to_csv(too_large_path, index=False)

    with pytest.raises(ValueError, match="exceed A-1"):
        from_edge_csv(too_large_path, A=2)

    self_loop = pd.DataFrame({"node1": [1], "node2": [1]})
    self_loop_path = tmp_path / "self_loop.csv"
    self_loop.to_csv(self_loop_path, index=False)

    with pytest.raises(ValueError, match="Self-loop detected"):
        from_edge_csv(self_loop_path, A=3)


def test_load_georgia_counties_without_path_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Georgia county adjacency graph not yet built"):
        load_georgia_counties()


def test_load_georgia_counties_with_edge_csv(tmp_path):
    # Minimal connected graph on 159 nodes: a chain
    node1 = np.arange(0, 158, dtype=int)
    node2 = np.arange(1, 159, dtype=int)
    df = pd.DataFrame({"node1": node1, "node2": node2})

    path = tmp_path / "ga_chain.csv"
    df.to_csv(path, index=False)

    graph = load_georgia_counties(path)

    assert graph.A == 159
    assert graph.n_edges == 158
    assert graph.name == "georgia_159_counties"
    assert graph.is_connected is True
    assert graph.scaling_factor > 0