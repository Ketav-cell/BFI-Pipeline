from __future__ import annotations
from typing import Dict, List, Tuple
import warnings
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_FRONTAL = [0, 1, 2, 3, 4, 5, 6]
_TEMPORAL = [10, 11, 12, 13]
_PARIETAL = [14, 15, 16]
_OCCIPITAL = [17, 18]
_CENTRAL = [7, 8, 9]
_THRESHOLD = cfg.NETWORK_THRESHOLD
_CH_IDX_MAP = {ch: i for i, ch in enumerate(cfg.STANDARD_CHANNELS)}
_CH_IDX = _CH_IDX_MAP
try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    warnings.warn('networkx not installed — network features will be 0.0.', ImportWarning)
try:
    import community as community_louvain
    _HAS_LOUVAIN = True
except ImportError:
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        _HAS_LOUVAIN = False
    except ImportError:
        _HAS_LOUVAIN = False
        warnings.warn('python-louvain not installed; modularity via nx fallback.', ImportWarning)

def _threshold_matrix(mat: np.ndarray, frac: float=_THRESHOLD) -> np.ndarray:
    n = mat.shape[0]
    upper = mat[np.triu_indices(n, k=1)]
    if len(upper) == 0:
        return mat.copy()
    cutoff = np.quantile(upper, 1.0 - frac)
    thresh = mat.copy()
    thresh[mat < cutoff] = 0.0
    return thresh

def _build_graph(adj: np.ndarray) -> 'nx.Graph':
    G = nx.Graph()
    G.add_nodes_from(range(adj.shape[0]))
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                G.add_edge(i, j, weight=float(adj[i, j]))
    return G

def _global_efficiency(G: 'nx.Graph') -> float:
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

def _modularity(G: 'nx.Graph', adj: np.ndarray) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    if _HAS_LOUVAIN:
        partition = community_louvain.best_partition(G, weight='weight')
        return float(community_louvain.modularity(partition, G, weight='weight'))
    else:
        try:
            communities = list(greedy_modularity_communities(G, weight='weight'))
            return float(nx.algorithms.community.quality.modularity(G, communities, weight='weight'))
        except Exception:
            return 0.0

def _betweenness(G: 'nx.Graph') -> np.ndarray:
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(n)
    bc = nx.betweenness_centrality(G, normalized=True, weight='weight')
    return np.array([bc.get(i, 0.0) for i in range(n)], dtype=np.float32)

def _network_variance(adj: np.ndarray) -> float:
    upper = adj[np.triu_indices(adj.shape[0], k=1)]
    return float(np.var(upper))

def _clustering(G: 'nx.Graph', nodes: List[int]) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    cc = nx.clustering(G, weight='weight')
    vals = [cc.get(n, 0.0) for n in nodes if n in G.nodes()]
    return float(np.mean(vals)) if vals else 0.0

def extract_network(wpli_matrix: np.ndarray) -> Dict[str, float]:
    if not _HAS_NX:
        return {'global_efficiency': 0.0, 'modularity': 0.0, 'bc_frontal': 0.0, 'bc_temporal': 0.0, 'bc_parietal': 0.0, 'bc_all_mean': 0.0, 'network_variance': 0.0, 'frontal_clustering': 0.0, 'posterior_clustering': 0.0, 'hemispheric_density': 0.0}
    adj = _threshold_matrix(wpli_matrix)
    G = _build_graph(adj)
    eff = _global_efficiency(G)
    mod = _modularity(G, adj)
    bc = _betweenness(G)
    bc_frontal = float(bc[_FRONTAL].mean()) if len(_FRONTAL) else 0.0
    bc_temporal = float(bc[_TEMPORAL].mean()) if len(_TEMPORAL) else 0.0
    bc_parietal = float(bc[_PARIETAL].mean()) if len(_PARIETAL) else 0.0
    frontal_clust = _clustering(G, _FRONTAL)
    posterior_clust = _clustering(G, _PARIETAL + _OCCIPITAL)
    left_nodes = [_CH_IDX[c] for c in cfg.LEFT_CHANNELS if c in _CH_IDX_MAP]
    right_nodes = [_CH_IDX[c] for c in cfg.RIGHT_CHANNELS if c in _CH_IDX_MAP]

    def _subgraph_density(nodes):
        sub = G.subgraph(nodes)
        n = len(nodes)
        if n < 2:
            return 0.0
        max_e = n * (n - 1) / 2
        return float(sub.number_of_edges() / max_e) if max_e > 0 else 0.0
    h_density = (_subgraph_density(left_nodes) + _subgraph_density(right_nodes)) / 2.0
    return {'global_efficiency': eff, 'modularity': mod, 'bc_frontal': bc_frontal, 'bc_temporal': bc_temporal, 'bc_parietal': bc_parietal, 'bc_all_mean': float(bc.mean()), 'network_variance': _network_variance(adj), 'frontal_clustering': frontal_clust, 'posterior_clustering': posterior_clust, 'hemispheric_density': h_density}

def network_feature_vector(wpli_matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    d = extract_network(wpli_matrix)
    names = sorted(d.keys())
    vec = np.array([d[k] for k in names], dtype=np.float32)
    return (vec, names)