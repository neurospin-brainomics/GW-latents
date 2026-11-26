"""Toy check: a Swiss roll and its unrolled version (the same 300 samples, the same intrinsic coordinates).

Straight-line distances cut across the layers of the roll; geodesic distances follow it. Expected (CPU, ~10 s):
  euclidean     contrast ~0.05, z ~1, ntrT ~1      -> nothing beyond the spectrum, no sample recovered
  geodesic_k10  contrast ~0.16, z ~25, ntrT ~4-7   -> shared shape beyond the spectrum, samples start to be recovered
The first grid point (1.5 eps_c) is a control: its contrast should be ~0 (its z is meaningless there).

    python examples/toy_swiss_roll.py
"""
import time

import numpy as np

import latent_gw as lg

rng = np.random.default_rng(0)
n = 300
t = 1.5 * np.pi * (1 + 2 * rng.random(n))                 # angle along the roll
h = 12 * rng.random(n)                                    # height
X = np.c_[t * np.cos(t), h, t * np.sin(t)] + 0.1 * rng.normal(size=(n, 3))     # rolled, 3-D
s = 0.5 * (t * np.sqrt(1 + t**2) + np.arcsinh(t))         # arc length along the spiral
Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
Y = np.c_[s, h, np.zeros((n, 3))] @ Q + 0.1 * rng.normal(size=(n, 5))          # unrolled, flat, other axes, 5-D

cols = ["rel", "rA", "rA_surr", "contrast", "z", "ntrT", "top1"]
for metric in ["euclidean", "geodesic_k10"]:
    t0 = time.time()
    res = lg.compare(X, Y, metric=metric, grid=(1.5, 0.3, 0.1, 0.03), surrogate="pcperm", B=3,
                     l2=False, dev="cpu")   # l2=False: projecting a roll on the sphere would distort it
    print(f"\n{metric}  ({time.time() - t0:.0f} s)\n" + res[cols].round(3).to_string(index=False))
