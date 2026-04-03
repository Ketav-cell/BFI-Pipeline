"""
features/network.py — Graph-theoretic network features from wPLI connectivity.

Pipeline per window:
  1. Receive wPLI matrix (n_channels × n_channels) from coordination.py
  2. Threshold: keep top NETWORK_THRESHOLD fraction of edges (by weight)
  3. Build undirected weighted graph (networkx)
  4. Extract:
       • Global efficiency     (inverse average shortest path)
       • Modularity            (Louvain — python-louvain / community package)
       • Betweenness centrality per node → frontal, temporal, parietal hub means

Channel groupings (indices into STANDARD_CHANNELS):
  Fp1(0)  Fp2(1)  F3(2)  F4(3)  F7(4)  F8(5)  Fz(6)
  C3(7)   C4(8)   Cz(9)
  T3(10)  T4(11)  T5(12) T6(13)
  P3(14)  P4(15)  Pz(16)
  O1(17)  O2(18)
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import warnings
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# Channel group indices
_FRONTAL   = [0, 1, 2, 3, 4, 5, 6]    # Fp1,Fp2,F3,F4,F7,F8,Fz
_TEMPORAL  = [10, 11, 12, 13]           # T3,T4,T5,T6
_PARIETAL  = [14, 15, 16]              # P3,P4,Pz
_OCCIPITAL = [17, 18]                  # O1,O2
_CENTRAL   = [7, 8, 9]                 # C3,C4,Cz

_THRESHOLD = cfg.NETWORK_THRESHOLD     # top 20% edges

# Channel index map (resolved at module load)
_CH_IDX_MAP = {ch: i for i, ch in enumerate(cfg.STANDARD_CHANNELS)}
_CH_IDX     = _CH_IDX_MAP

# ─── Optional imports ─────────────────────────────────────────────────────────

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    warnings.warn("networkx not installed — network features will be 0.0.", ImportWarning)

try:
    import community as community_louvain
    _HAS_LOUVAIN = True
except ImportError:
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        _HAS_LOUVAIN = False   # use networkx fallback
    except ImportError:
        _HAS_LOUVAIN = False
        warnings.warn("python-louvain not installed; modularity via nx fallback.", ImportWarning)


# ─── Thresholding ─────────────────────────────────────────────────────────────

def _threshold_matrix(mat: np.ndarray, frac: float = _THRESHOLD) -> np.ndarray:
    """Zero out all but the top *frac* fraction of edges."""
    n = mat.shape[0]
    upper = mat[np.triu_indices(n, k=1)]
    if len(upper) == 0:
        return mat.copy()
    cutoff = np.quantile(upper, 1.0 - frac)
    thresh = mat.copy()
    thresh[mat < cutoff] = 0.0
    return thresh


# ─── Graph metrics ────────────────────────────────────────────────────────────

def _build_graph(adj: np.ndarray) -> "nx.Graph":
    G = nx.Graph()
    G.add_nodes_from(range(adj.shape[0]))
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                G.add_edge(i, j, weight=float(adj[i, j]))
    return G


def _global_efficiency(G: "nx.Graph") -> float:
    """
    E_glob = 1/n * Σ_i Σ_{j≠i} 1/d(i,j)
    Disconnected pairs contribute 0 (i.e., 1/inf = 0).
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    total = 0.0
    for node in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, node)
        for other, dist in lengths.items():
            if other != node and dist > 0:
                total += 1.0 / dist
    return float(total / (n * (n - 1)))


def _modularity(G: "nx.Graph", adj: np.ndarray) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    if _HAS_LOUVAIN:
        partition = community_louvain.best_partition(G, weight="weight")
        return float(community_louvain.modularity(partition, G, weight="weight"))
    else:
        # NetworkX greedy fallback
        try:
            communities = list(greedy_modularity_communities(G, weight="weight"))
            return float(nx.algorithms.community.quality.modularity(
                G, communities, weight="weight"
            ))
        except Exception:
            return 0.0


def _betweenness(G: "nx.Graph") -> np.ndarray:
    """Return betweenness centrality as array indexed by node."""
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(n)
    bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
    return np.array([bc.get(i, 0.0) for i in range(n)], dtype=np.float32)


# ─── Network variance ─────────────────────────────────────────────────────────

def _network_variance(adj: np.ndarray) -> float:
    upper = adj[np.triu_indices(adj.shape[0], k=1)]
    return float(np.var(upper))


# ─── Clustering coefficient ───────────────────────────────────────────────────

def _clustering(G: "nx.Graph", nodes: List[int]) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    cc = nx.clustering(G, weight="weight")
    vals = [cc.get(n, 0.0) for n in nodes if n in G.nodes()]
    return float(np.mean(vals)) if vals else 0.0


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_network(wpli_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute graph-theoretic features from the wPLI connectivity matrix.

    Parameters
    ----------
    wpli_matrix : (n_channels, n_channels) symmetric float array

    Returns
    -------
    dict of scalar feature values.
    """
    if not _HAS_NX:
        return {
            "global_efficiency":     0.0,
            "modularity":            0.0,
            "bc_frontal":            0.0,
            "bc_temporal":           0.0,
            "bc_parietal":           0.0,
            "bc_all_mean":           0.0,
            "network_variance":      0.0,
            "frontal_clustering":    0.0,
            "posterior_clustering":  0.0,
            "hemispheric_density":   0.0,
        }

    adj = _threshold_matrix(wpli_matrix)
    G   = _build_graph(adj)

    eff = _global_efficiency(G)
    mod = _modularity(G, adj)
    bc  = _betweenness(G)

    bc_frontal  = float(bc[_FRONTAL].mean())   if len(_FRONTAL)   else 0.0
    bc_temporal = float(bc[_TEMPORAL].mean())  if len(_TEMPORAL)  else 0.0
    bc_parietal = float(bc[_PARIETAL].mean())  if len(_PARIETAL)  else 0.0

    # Clustering
    frontal_clust   = _clustering(G, _FRONTAL)
    posterior_clust = _clustering(G, _PARIETAL + _OCCIPITAL)

    # Hemispheric density: fraction of edges present in left vs right
    left_nodes  = [_CH_IDX[c] for c in cfg.LEFT_CHANNELS  if c in _CH_IDX_MAP]
    right_nodes = [_CH_IDX[c] for c in cfg.RIGHT_CHANNELS if c in _CH_IDX_MAP]

    def _subgraph_density(nodes):
        sub = G.subgraph(nodes)
        n   = len(nodes)
        if n < 2:
            return 0.0
        max_e = n * (n - 1) / 2
        return float(sub.number_of_edges() / max_e) if max_e > 0 else 0.0

    h_density = (_subgraph_density(left_nodes) + _subgraph_density(right_nodes)) / 2.0

    return {
        "global_efficiency":    eff,
        "modularity":           mod,
        "bc_frontal":           bc_frontal,
        "bc_temporal":          bc_temporal,
        "bc_parietal":          bc_parietal,
        "bc_all_mean":          float(bc.mean()),
        "network_variance":     _network_variance(adj),
        "frontal_clustering":   frontal_clust,
        "posterior_clustering": posterior_clust,
        "hemispheric_density":  h_density,
    }


def network_feature_vector(wpli_matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    d     = extract_network(wpli_matrix)
    names = sorted(d.keys())
    vec   = np.array([d[k] for k in names], dtype=np.float32)
    return vec, names
