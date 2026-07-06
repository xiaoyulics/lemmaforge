"""forge.lp_check -- S3b sanity harness: discretized primal LP.

Independent numerical cross-check of every univariate cell: discretize the
support on a window inferred from the moment constraints, solve the primal
LP over atom weights with scipy.linprog (HiGHS), compare with the SDP value.
Disagreement > 1e-4 relative is an open issue (PLAN S3b).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .dsl import LemmaInstance

import sympy as sp

X = sp.Symbol("x", real=True)


def lp_value(inst: LemmaInstance, n_grid: int = 20001, window: float | None = None) -> float:
    """Discretized primal value (sup or inf) of the univariate cell."""
    # window from second/fourth moment scale, generous
    if window is None:
        scale = 1.0
        for c in inst.constraints:
            d = sp.degree(c.poly, X)
            if d >= 2 and c.value > 0:
                scale = max(scale, float(c.value) ** (1.0 / float(d)))
        tval = float(inst.event_t) if inst.event_t is not None else 0.0
        window = max(12.0 * scale, 3.0 * abs(tval) + 6.0)

    lo = 0.0 if inst.support == "half_line" and inst.support_a == 0 else -window
    if inst.support == "half_line":
        lo = float(inst.support_a)
    hi = window
    if inst.support == "interval":
        lo, hi = float(inst.support_a), float(inst.support_b)
    xs = np.linspace(lo, hi, n_grid)
    # geometric far tail: moment-cone-boundary cells need escaping mass
    # (an atom at M with mass ~ c/M^d); a linear grid truncates it. Cap so
    # that polynomial values stay within HiGHS-friendly magnitude.
    Dmax = max(int(sp.degree(c.poly, X)) for c in inst.constraints)
    cap = 10.0 ** (10.0 / max(Dmax, 1))
    far = hi * np.power(1.25, np.arange(1, 40))
    far = far[far <= max(cap, 2 * hi)]
    xs = np.concatenate([xs, far])
    if lo < 0:
        xs = np.concatenate([xs, -far])
    # make sure the event threshold and other structurally-special points are
    # grid points (extremal atoms often sit exactly there)
    extra = [1.0, -1.0]
    if inst.event_t is not None:
        t = float(inst.event_t)
        extra += [t, -t]
    xs = np.append(xs, [p for p in extra if lo <= p <= hi + 1e-12])
    xs = np.sort(np.unique(xs))

    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    for c in inst.constraints:
        f = sp.lambdify(X, c.poly, "numpy")
        row = np.asarray(f(xs), dtype=float) * np.ones(len(xs))
        if c.op == "==":
            A_eq.append(row); b_eq.append(float(c.value))
        elif c.op == "<=":
            A_ub.append(row); b_ub.append(float(c.value))
        else:
            A_ub.append(-row); b_ub.append(-float(c.value))

    if inst.obj_type == "probability":
        t = float(inst.event_t)
        cvec = (xs >= t - 1e-12).astype(float)
    else:
        f = sp.lambdify(X, inst.obj_poly, "numpy")
        cvec = np.asarray(f(xs), dtype=float) * np.ones(len(xs))

    sign = -1.0 if inst.sense == "sup" else 1.0
    res = linprog(sign * cvec,
                  A_eq=np.array(A_eq) if A_eq else None,
                  b_eq=np.array(b_eq) if b_eq else None,
                  A_ub=np.array(A_ub) if A_ub else None,
                  b_ub=np.array(b_ub) if b_ub else None,
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return float(sign * res.fun) if inst.sense == "sup" else float(res.fun)
