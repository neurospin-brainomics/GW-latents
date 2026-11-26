"""Quickstart: is a raw GW score evidence of shared shape?

Three clouds of 200 points in 8 dimensions with the same anisotropy (variance spectrum):
X, a rotated noisy copy of X, and an independent cloud with the same spectrum.
"""
import numpy as np

import latent_gw as lg

rng = np.random.default_rng(0)
spectrum = np.linspace(2, 0.3, 8)
X = rng.normal(size=(200, 8)) * spectrum
Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
cases = {"related": X @ Q + 0.1 * rng.normal(size=X.shape),
         "unrelated, same spectrum": rng.normal(size=(200, 8)) * spectrum}

cols = ["rel", "rA", "rA_surr", "contrast", "z", "ntrT", "cka"]
for name, Y in cases.items():
    res = lg.compare(X, Y, metric="euclidean", grid=(1.5, 0.3, 0.1, 0.03), surrogate="pcperm", B=8)
    print(f"\n{name}\n" + res[cols].round(3).to_string(index=False))
