"""forge.recognize -- S4: exact-value recognition from solver floats.

Outputs are conjectures labeled RECOGNIZED, never proofs (PLAN 2.6).
Confidence protocol: a candidate must reproduce the float to the working
precision AND remain stable when re-checked at doubled precision when a
high-precision value is available (for SDP output we only have ~1e-9, so
the confidence field records the fit residual and denominator size).
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import sympy as sp


def recognize_value(v: float, tol: float = 5e-9, max_den: int = 10**6) -> dict | None:
    """Try rational, then quadratic surd. Return conjecture dict or None."""
    # 1) rational with modest denominator
    fr = Fraction(v).limit_denominator(max_den)
    if abs(float(fr) - v) < tol:
        return {"kind": "rational", "expr": sp.Rational(fr.numerator, fr.denominator),
                "residual": abs(float(fr) - v)}
    # 2) quadratic algebraic via PSLQ on [1, v, v^2]
    mp.mp.dps = 30
    rel = mp.pslq([mp.mpf(1), mp.mpf(v), mp.mpf(v) ** 2], maxcoeff=10**8, maxsteps=10**4)
    if rel and rel[2] != 0:
        c0, c1, c2 = [int(r) for r in rel]
        x = sp.Symbol("x")
        for root in sp.solve(c0 + c1 * x + c2 * x**2, x):
            if abs(complex(root.evalf()).imag) < 1e-20 and abs(float(root) - v) < tol:
                return {"kind": "quadratic", "expr": sp.nsimplify(root),
                        "residual": abs(float(root) - v),
                        "minpoly": (c0, c1, c2)}
    return None


def fit_sequence(ns: list[int], vs: list[float], tol: float = 1e-7) -> dict | None:
    """Fit v(n) against rational-function bases of increasing complexity,
    using exact linear algebra on rationalized values. Returns the simplest
    basis that fits ALL points within tol. This is the module that must
    autonomously produce 3 - 2/n from the B5 numerics (PLAN Phase 1)."""
    n = sp.Symbol("n")
    bases = [
        [sp.Integer(1)],
        [sp.Integer(1), 1 / n],
        [sp.Integer(1), 1 / n, 1 / n**2],
        [sp.Integer(1), 1 / n, 1 / sp.sqrt(n)],
        [sp.Integer(1), n],
    ]
    pts = [(sp.Integer(k), sp.Rational(Fraction(v).limit_denominator(10**9)))
           for k, v in zip(ns, vs)]
    for basis in bases:
        k = len(basis)
        if len(pts) < k + 1:      # demand at least one out-of-sample check
            continue
        A = sp.Matrix([[b.subs(n, kk) for b in basis] for kk, _ in pts[:k]])
        y = sp.Matrix([vv for _, vv in pts[:k]])
        try:
            coef = A.solve(y)
        except Exception:
            continue
        # snap coefficients to small rationals (they came from noisy floats)
        coef = sp.Matrix([sp.nsimplify(c, rational=True, tolerance=1e-6) for c in coef])
        expr = sp.expand(sum(c * b for c, b in zip(coef, basis)))
        ok = all(abs(float(expr.subs(n, kk)) - float(vv)) < tol for kk, vv in pts)
        if ok:
            return {"kind": "sequence", "expr": sp.nsimplify(expr),
                    "basis": [sp.sstr(b) for b in basis],
                    "max_residual": max(abs(float(expr.subs(n, kk)) - float(vv))
                                        for kk, vv in pts)}
    return None
