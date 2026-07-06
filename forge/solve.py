"""forge.solve -- solver driver (S3) with residual reporting and provenance."""

from __future__ import annotations

import datetime
import subprocess
import time

import cvxpy as cp
import numpy as np

from .compile_sdp import CompiledSDP, pseudo_moments


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def solve(sdp: CompiledSDP, solver_chain: tuple[str, ...] = ("CLARABEL", "SCS")) -> dict:
    """Solve; return record with value, lambda, gram blocks, moments, residuals."""
    prob = sdp.problem
    last_err = None
    used = None
    t0 = time.time()
    for name in solver_chain:
        try:
            kwargs = {"solver": name}
            if name == "CLARABEL":
                kwargs.update(dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11,
                                   tol_feas=1e-11))
            elif name == "SCS":
                kwargs.update(dict(eps=1e-9, max_iters=200000))
            prob.solve(**kwargs)
            if prob.status in ("optimal", "optimal_inaccurate"):
                used = name
                break
        except (cp.SolverError, Exception) as e:  # noqa: BLE001
            last_err = e
    wall = time.time() - t0
    if used is None:
        return {"status": "SOLVER_FAILED", "error": str(last_err)}

    lam = np.asarray(sdp.lam.value, dtype=float)
    grams = []
    max_min_eig = 0.0
    min_eig_all = np.inf
    for piece in sdp.pieces:
        for blk in piece.blocks:
            G = np.asarray(blk.var.value, dtype=float)
            G = 0.5 * (G + G.T)
            ev = float(np.linalg.eigvalsh(G).min()) if G.size else 0.0
            min_eig_all = min(min_eig_all, ev)
            grams.append({"piece": piece.role, "multiplier": str(blk.multiplier),
                          "basis": [str(b) for b in blk.basis], "G": G, "min_eig": ev})

    # residuals of the coefficient equalities
    res = 0.0
    for c in sdp.coeff_constraints:
        v = c.violation()
        res = max(res, float(np.max(np.abs(v))) if np.ndim(v) else abs(float(v)))

    return {
        "status": prob.status,
        "value": float(prob.value),
        "lambda": lam,
        "grams": grams,
        "moments": pseudo_moments(sdp),
        "max_residual": res,
        "min_gram_eig": float(min_eig_all),
        "provenance": {
            "solver": f"{used} via cvxpy {cp.__version__}",
            "git": _git_hash(),
            "wall_time_s": round(wall, 3),
            "date": datetime.date.today().isoformat(),
            "residual": res,
        },
    }
