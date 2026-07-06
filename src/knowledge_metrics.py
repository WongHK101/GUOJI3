"""
Knowledge Reliability Metrics for ISTF-JRNGC (KBS Phase 3.1).

Convention:
  - Score matrix S has shape (d, d) where S[target, source] = GC score of source->target.
  - Edges are returned as (source, target) tuples.
  - Tie-breaking key for exact top-k: (-score, target_idx, source_idx).
    Highest score wins; tied on score -> lower target index wins;
    tied on target -> lower source index wins.
  - All binarization-dependent metrics call the same topk_edges_exact() helper.
  - No local threshold, no np.sort, no scores >= threshold anywhere.

PKD formula:
  PKD = max(0, (loss_b - loss_m) / (abs(loss_b) + eps)) * max(0, quality_b - quality_m)
  PKD is a risk index, not a performance metric.
  PKD > 0  -> decoupling detected (prediction improved but knowledge degraded).
  PKD = 0  -> no decoupling (aligned or one term non-positive).
"""

import warnings
import numpy as np

_EPS = 1e-12
_EPS_DIV = 1e-8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kf_aggregate(arr):
    """Max-absolute over lag dimension, consistent with benchmark protocol.

    If arr.ndim == 3, returns np.max(np.abs(arr), axis=2).
    If arr.ndim == 2, returns np.abs(arr).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        d0, d1 = arr.shape[0], arr.shape[1]
        if d0 != d1:
            raise ValueError(f"Score matrix first two dims must be square, got ({d0},{d1})")
        return np.max(np.abs(arr), axis=2)
    elif arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Score matrix must be square, got shape {arr.shape}")
        return np.abs(arr)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def _to_2d_adj(arr):
    """Convert (d,d) or (d,d,lag) adjacency to (d,d) bool.

    Raises ValueError if first two dims are not square.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Adjacency must be square 2D, got shape {arr.shape}")
        return arr.astype(bool)
    if arr.ndim == 3:
        d0, d1 = arr.shape[0], arr.shape[1]
        if d0 != d1:
            raise ValueError(f"Adjacency first two dims must be square, got ({d0},{d1})")
        if arr.shape[2] == 1:
            return arr[:, :, 0].astype(bool)
        return np.any(arr, axis=2)
    raise ValueError(f"Expected 2D or 3D adjacency, got shape {arr.shape}")


def adjacency_to_edge_set(adj_2d, exclude_diag=True):
    """Convert adjacency matrix A[target, source] to set of (source, target) tuples.

    Args:
        adj_2d: (d, d) square binary adjacency matrix.
        exclude_diag: If True, self-loops (target == source) are excluded.

    Returns:
        set of (source_idx, target_idx) tuples for each True entry.
    """
    adj = np.asarray(adj_2d).astype(bool)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square 2D matrix, got shape {adj.shape}")
    d = adj.shape[0]
    edges = set()
    for target in range(d):
        for source in range(d):
            if exclude_diag and target == source:
                continue
            if adj[target, source]:
                edges.add((source, target))
    return edges


# ---------------------------------------------------------------------------
# Exact top-k edge selection
# ---------------------------------------------------------------------------

def topk_edges_exact(scores_2d, k, exclude_diag=True):
    """Return exactly k directed edges with deterministic tie-breaking.

    Args:
        scores_2d: (d, d) array where entry [tgt, src] is the score of src->tgt.
        k: Exact number of edges to return.
        exclude_diag: If True, self-loops are not eligible.

    Returns:
        set of (source_idx, target_idx) tuples, size exactly min(k, d*(d-1)).

    Tie-breaking: candidates are sorted by (-score, target_idx, source_idx).
    This guarantees deterministic output even when scores are tied.
    """
    scores_2d = np.asarray(scores_2d, dtype=np.float64)
    if scores_2d.ndim != 2 or scores_2d.shape[0] != scores_2d.shape[1]:
        raise ValueError(f"scores_2d must be square 2D matrix, got shape {scores_2d.shape}")

    d = scores_2d.shape[0]

    if k <= 0:
        return set()

    candidates = []
    for target in range(d):
        for source in range(d):
            if exclude_diag and target == source:
                continue
            score = float(scores_2d[target, source])
            candidates.append((-score, target, source))

    candidates.sort()
    max_k = len(candidates)
    if k > max_k:
        warnings.warn(
            f"topk_edges_exact: k={k} exceeds max candidates={max_k} (d={d}). "
            f"Clamping to {max_k}."
        )
        k = max_k

    return {(source, target) for _, target, source in candidates[:k]}


# ---------------------------------------------------------------------------
# KF — Knowledge Fidelity
# ---------------------------------------------------------------------------

def compute_kf_maxlag(gc_scores, true_coefficients):
    """Knowledge Fidelity with max-lag aggregation.

    Computes Pearson correlation between off-diagonal absolute GC scores
    and true coefficient magnitudes, both max-aggregated over lag dimension.

    Args:
        gc_scores: (d, d) or (d, d, lag) predicted GC scores.
        true_coefficients: (d, d) or (d, d, lag) true generative coefficients
            (NOT binary adjacency — must be real-valued coefficient matrix).

    Returns:
        float in [-1, 1], or None if correlation cannot be computed
        (zero variance, shape mismatch, or no valid off-diagonal entries).

    Applicable only to datasets with explicit known generative coefficients
    (VAR, NSVAR, synthetic). For binary ground-truth datasets, returns None.
    """
    try:
        scores_2d = _kf_aggregate(gc_scores)
        coef_2d = _kf_aggregate(true_coefficients)
    except (ValueError, TypeError):
        return None

    if scores_2d.shape != coef_2d.shape:
        return None

    d = scores_2d.shape[0]
    mask = ~np.eye(d, dtype=bool)
    scores_off = scores_2d[mask]
    coef_off = coef_2d[mask]

    if scores_off.size == 0:
        return None
    if np.std(scores_off) < _EPS or np.std(coef_off) < _EPS:
        return None

    corr = np.corrcoef(scores_off, coef_off)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


# ---------------------------------------------------------------------------
# FKR — False Knowledge Rate
# ---------------------------------------------------------------------------

def compute_fkr(gc_scores, true_edges, k=None):
    """False Knowledge Rate at exact top-k.

    FKR@k = |pred_edges \\ true_edges| / |pred_edges|
    Proportion of extracted causal edges that are false.

    Args:
        gc_scores: (d, d) or (d, d, lag) predicted GC scores.
        true_edges: (d, d) or (d, d, lag) binary ground-truth adjacency.
        k: Number of top edges. Defaults to |E*| (unique off-diagonal true edges).

    Returns:
        float in [0, 1]. k=0 returns 0.0.
    """
    true_adj = _to_2d_adj(true_edges)
    true_edge_set = adjacency_to_edge_set(true_adj, exclude_diag=True)
    n_edges_true = len(true_edge_set)

    if k is None:
        k = n_edges_true
    if k <= 0:
        return 0.0

    scores_2d = _kf_aggregate(gc_scores)
    pred_edges = topk_edges_exact(scores_2d, k, exclude_diag=True)
    false_edges = pred_edges - true_edge_set
    denom = max(len(pred_edges), 1)
    return len(false_edges) / denom


# ---------------------------------------------------------------------------
# MKR — Missing Knowledge Rate
# ---------------------------------------------------------------------------

def compute_mkr(gc_scores, true_edges, k=None):
    """Missing Knowledge Rate at exact top-k.

    MKR@k = |true_edges \\ pred_edges| / |E*|
    Proportion of true causal edges missed in the top-k.

    Args:
        gc_scores: (d, d) or (d, d, lag) predicted GC scores.
        true_edges: (d, d) or (d, d, lag) binary ground-truth adjacency.
        k: Number of top edges. Defaults to |E*| (unique off-diagonal true edges).

    Returns:
        float in [0, 1]. |E*|=0 returns 0.0.
    """
    true_adj = _to_2d_adj(true_edges)
    true_edge_set = adjacency_to_edge_set(true_adj, exclude_diag=True)
    n_edges_true = len(true_edge_set)

    if k is None:
        k = n_edges_true
    if n_edges_true <= 0:
        return 0.0

    scores_2d = _kf_aggregate(gc_scores)
    pred_edges = topk_edges_exact(scores_2d, k, exclude_diag=True)
    missing_edges = true_edge_set - pred_edges
    return len(missing_edges) / max(n_edges_true, 1)


# ---------------------------------------------------------------------------
# KS — Knowledge Stability
# ---------------------------------------------------------------------------

def compute_ks(gc_scores_list, true_edges=None, k=None):
    """Knowledge Stability: mean pairwise Jaccard of exact top-k edge sets.

    KS = (2 / (R*(R-1))) * sum_{a<b} Jaccard(E_a, E_b)

    KS measures cross-seed reproducibility of top-k edge sets.
    KS does NOT imply correctness. High KS with high FKR means
    stably wrong — the method consistently extracts the same wrong edges.

    Convention: when k=0 and R>=2, all top-k sets are empty and
    Jaccard = 1.0 for each pair (all seeds agree on "no edges").

    Args:
        gc_scores_list: List of (d,d) or (d,d,lag) score arrays, one per seed.
        true_edges: (d, d) or (d, d, lag) binary ground-truth adjacency. Only used to
            infer default k = |E*| (unique off-diagonal true edges).
        k: Explicit number of top edges. Required if true_edges is None.

    Returns:
        float in [0, 1], or None if R < 2.
    """
    R = len(gc_scores_list)
    if R < 2:
        return None

    if true_edges is not None:
        true_adj = _to_2d_adj(true_edges)
        true_edge_set = adjacency_to_edge_set(true_adj, exclude_diag=True)
        if k is None:
            k = len(true_edge_set)
    else:
        if k is None:
            raise ValueError("compute_ks: must provide either true_edges or explicit k")

    edge_sets = []
    for scores in gc_scores_list:
        scores_2d = _kf_aggregate(scores)
        edge_set = topk_edges_exact(scores_2d, k, exclude_diag=True)
        edge_sets.append(edge_set)

    jaccards = []
    for a in range(R):
        for b in range(a + 1, R):
            union = len(edge_sets[a] | edge_sets[b])
            inter = len(edge_sets[a] & edge_sets[b])
            if union > 0:
                jaccards.append(inter / union)
            else:
                jaccards.append(1.0)

    return float(np.mean(jaccards))


# ---------------------------------------------------------------------------
# PKD — Prediction–Knowledge Decoupling Index
# ---------------------------------------------------------------------------

def compute_pkd(loss_baseline, loss_method, quality_baseline, quality_method, eps=_EPS_DIV):
    """Prediction–Knowledge Decoupling index (risk index, not performance metric).

    PKD = max(0, (loss_b - loss_m) / (abs(loss_b) + eps)) * max(0, quality_b - quality_m)

    Interpretation:
      PKD > 0 : decoupling detected — prediction improved but knowledge degraded.
      PKD = 0 : no decoupling — aligned, or one term non-positive.

    This is a risk index only. It must NOT be interpreted as a performance score.

    Args:
        loss_baseline: Prediction loss of baseline method (lower is better).
        loss_method: Prediction loss of method being evaluated.
        quality_baseline: Knowledge quality of baseline (higher is better).
        quality_method: Knowledge quality of method being evaluated.
        eps: Small constant to prevent division by zero.

    Returns:
        float >= 0.0.
    """
    delta_loss_norm = max(0.0, (loss_baseline - loss_method) / (abs(loss_baseline) + eps))
    delta_quality = max(0.0, quality_baseline - quality_method)
    return delta_loss_norm * delta_quality
