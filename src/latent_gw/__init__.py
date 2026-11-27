"""latent_gw: compare latent spaces with entropic Gromov–Wasserstein.

The core functions are taken unchanged from the research scripts that produced the results
(synth_graph_metrics.py, ricci_flow.py, 178_r_null_test.py, 199_bifurcation_real.py, 215_champ_map.py,
224_gw_story.py); `compare` is the per-pair computation of the Champollion maps.
"""
from .cka import cka_u, linear_cka
from .cka_map import cka_matrix, homologue_test, subsample_index
from .compare import compare, path, prep_region
from .curvature import orc_flow_distance
from .distances import dist_from_emb, parse_metric, raw_distance
from .graphs import all_pairs_shortest_paths, connected_knn_graph
from .inference import mc_test, pair_test, pairing_tests
from .preprocessing import l2_points, rms
from .scores import ceiling, plan_stats
from .solver import describe, eps_c, gw_solve, run_path, solve
from .surrogates import (make_surrogate, mds_embedding, surr_elliptic, surr_native, surr_pcperm,
                         surr_spectral, surr_spectral_ecc)

# figures: `from latent_gw import plots` (needs matplotlib)
__version__ = "0.1.0"
