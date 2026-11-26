"""Distance matrices of a cloud: Euclidean, geodesic kNN, Ollivier–Ricci (adapted from 178_r_null_test.dist_from_emb).

Metric names: "euclidean", "geodesic_k<k>" (e.g. geodesic_k20), "orc_k<k>" (one Ollivier–Ricci re-weighting),
"orc_k<k>_it<t>" (t steps of discrete Ricci flow).
Differences with the original dist_from_emb: no module-level L2 flag (passed as an argument), and the geodesic
metric always uses graphs.connected_knn_graph (178 used graph_metrics.py when importable; both implement the same
graph: symmetric kNN, k increased up to 2k+40, then MST bridging).
"""
import os
import re

import numpy as np
import torch

from . import curvature, graphs
from .preprocessing import DT, l2_points, rms


def parse_metric(name):
    """'euclidean' -> ('euclidean', None, 1); 'geodesic_k20' -> ('geodesic', 20, 1); 'orc_k20_it3' -> ('orc', 20, 3)."""
    if name == "euclidean":
        return "euclidean", None, 1
    m = re.fullmatch(r"(geodesic|orc)_k(\d+)(?:_it(\d+))?", name)
    if not m:
        raise ValueError(f"unknown metric {name!r}")
    return m.group(1), int(m.group(2)), int(m.group(3) or 1)


def raw_distance(X, metric, n_jobs=None):
    """Unnormalised distance matrix (numpy, float64)."""
    n_jobs = n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    kind, k, iters = parse_metric(metric)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if kind == "euclidean":
        return graphs.euclidean(X)
    if kind == "geodesic":
        G, *_ = graphs.connected_knn_graph(X, k)
        return graphs.all_pairs_shortest_paths(G, n_jobs)
    return curvature.orc_flow_distance(X, k, iters=iters, n_jobs=n_jobs)[0]


def dist_from_emb(X, metric, dev="cpu", l2=False, n_jobs=None):
    """RMS-normalised distance matrix (torch float64 on `dev`), optionally after L2 normalisation of the points."""
    if l2:
        X = l2_points(X)
    if metric == "euclidean":
        X = torch.as_tensor(X, dtype=DT, device=dev) if not torch.is_tensor(X) else X.to(dev, DT)
        return rms(torch.cdist(X, X))
    X = np.asarray(X.cpu().numpy() if torch.is_tensor(X) else X, dtype=np.float64)
    return rms(torch.as_tensor(raw_distance(X, metric, n_jobs), dtype=DT, device=dev))
