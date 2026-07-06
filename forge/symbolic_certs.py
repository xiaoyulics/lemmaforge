"""forge.symbolic_certs -- exact certificate constructors from the closed
forms proven in the paper (Theorem 3.1 and the classical benchmarks).

These bypass numerical rounding entirely: every entry is an exact sympy
expression (rational whenever the parameters make it so). The independent
verifier remains the sole authority (PD1) -- these constructors PRODUCE,
never certify.

All certificates are in forge/verify.py schema. Extremal witnesses included
where the optimum is attained.
"""

from __future__ import annotations

import sympy as sp

X = sp.Symbol("x", real=True)


def _s(e) -> str:
    return sp.sstr(sp.nsimplify(sp.expand(e)))


def _cons(polys_ops_vals) -> list[dict]:
    return [{"poly": _s(p), "op": o, "value": _s(v)} for p, o, v in polys_ops_vals]


def markov_cert(t) -> dict:
    t = sp.nsimplify(t)
    assert t >= 1
    V = 1 / t
    return {
        "lemma_id": "markov", "params": {"t": _s(t)},
        "claim": f"P(x >= {t}) <= {_s(V)} over {{E1=1, Ex=1}}, support [0,oo)",
        "objective": {"sense": "sup", "type": "probability"},
        "constraints": _cons([(sp.Integer(1), "==", 1), (X, "==", 1)]),
        "dual_multipliers": ["0", _s(1 / t)],
        "q_poly": _s(X / t), "bound_value": _s(V),
        "pieces": [
            {"role": "event", "domain": {"type": "half_line_ge", "a": _s(t)},
             "blocks": [{"multiplier": _s(X - t), "basis": ["1"], "gram": [[_s(1 / t)]]}]},
            {"role": "support", "domain": {"type": "half_line_ge", "a": "0"},
             "blocks": [{"multiplier": "x", "basis": ["1"], "gram": [[_s(1 / t)]]}]},
        ],
        "extremal": {"atoms": ["0", _s(t)], "weights": [_s(1 - V), _s(V)]},
        "status": "SYMBOLIC-UNVERIFIED",
        "provenance": {"source": "closed form (Markov, degree-1 exact)", "date": "2026-07-02"},
    }


def cantelli_cert(sigma2, t) -> dict:
    s2, t = sp.nsimplify(sigma2), sp.nsimplify(t)
    D = (t + s2 / t) ** 2
    V = s2 / (s2 + t ** 2)
    lam = [s2 ** 2 / t ** 2 / D, 2 * (s2 / t) / D, 1 / D]
    return {
        "lemma_id": "cantelli", "params": {"sigma2": _s(s2), "t": _s(t)},
        "claim": f"P(x >= {t}) <= {_s(V)} over {{E1=1, Ex=0, Ex2={_s(s2)}}}, support R",
        "objective": {"sense": "sup", "type": "probability"},
        "constraints": _cons([(sp.Integer(1), "==", 1), (X, "==", 0), (X ** 2, "==", s2)]),
        "dual_multipliers": [_s(v) for v in lam],
        "q_poly": _s((X + s2 / t) ** 2 / D), "bound_value": _s(V),
        "pieces": [
            {"role": "event", "domain": {"type": "half_line_ge", "a": _s(t)},
             "blocks": [
                 {"multiplier": "1", "basis": [_s(X - t)], "gram": [[_s(1 / D)]]},
                 {"multiplier": _s(X - t), "basis": ["1"],
                  "gram": [[_s(2 * (t + s2 / t) / D)]]}]},
            {"role": "support", "domain": {"type": "real_line"},
             "blocks": [{"multiplier": "1", "basis": ["1", "x"],
                         "gram": [[_s(s2 ** 2 / t ** 2 / D), _s(s2 / t / D)],
                                  [_s(s2 / t / D), _s(1 / D)]]}]},
        ],
        "extremal": {"atoms": [_s(t), _s(-s2 / t)], "weights": [_s(V), _s(1 - V)]},
        "status": "SYMBOLIC-UNVERIFIED",
        "provenance": {"source": "closed form (Cantelli / PLAN Appendix A)", "date": "2026-07-02"},
    }


def paley_zygmund_cert(m2, theta) -> dict:
    m2, th = sp.nsimplify(m2), sp.nsimplify(theta)
    b = (m2 - th) / (1 - th)
    V = (1 - th) ** 2 / ((1 - th) ** 2 + m2 - 1)
    B = (b - th) ** 2
    # q = 1 - (x-b)^2/B ; lambda from expansion: q = (B - b^2)/B + (2b/B) x - x^2/B
    lam = [(B - b ** 2) / B, 2 * b / B, -1 / B]
    return {
        "lemma_id": "paley_zygmund", "params": {"m2": _s(m2), "theta": _s(th)},
        "claim": f"P(x >= {th}) >= {_s(V)} over {{E1=1, Ex=1, Ex2={_s(m2)}}}, support [0,oo) "
                 f"(tight PZ; infimum, approached but not attained)",
        "objective": {"sense": "inf", "type": "probability"},
        "constraints": _cons([(sp.Integer(1), "==", 1), (X, "==", 1), (X ** 2, "==", m2)]),
        "dual_multipliers": [_s(v) for v in lam],
        "q_poly": _s(1 - (X - b) ** 2 / B), "bound_value": _s(V),
        "pieces": [
            {"role": "event", "domain": {"type": "half_line_ge", "a": _s(th)},
             "blocks": [{"multiplier": "1", "basis": [_s(X - b)], "gram": [[_s(1 / B)]]}]},
            {"role": "offevent", "domain": {"type": "interval", "a": "0", "b": _s(th)},
             "blocks": [
                 {"multiplier": "1", "basis": [_s(th - X)],
                  "gram": [[_s((2 * b - th) / (th * B))]]},
                 {"multiplier": _s(sp.expand(X * (th - X))), "basis": ["1"],
                  "gram": [[_s((2 * b - 2 * th) / (th * B))]]}]},
        ],
        "status": "SYMBOLIC-UNVERIFIED",
        "provenance": {"source": "closed form (tight Paley-Zygmund; memo b4_tight_pz)",
                       "date": "2026-07-02"},
    }


def kurtosis_regime(t, kappa) -> str:
    t, k = sp.nsimplify(t), sp.nsimplify(kappa)
    if k == 1:
        return "corner"
    b = (sp.sqrt(k + 3) - sp.sqrt(k - 1)) / 2
    c = (sp.sqrt(k + 3) + sp.sqrt(k - 1)) / 2
    if t >= c:
        return "II"
    if t >= b:
        return "I"
    # t < b: IIIa needs kappa <= 3/2 and t >= tau
    if k <= sp.Rational(3, 2):
        u, s = sp.sqrt(k - 1), sp.sqrt(k + 3)
        tau = -b + sp.sqrt(u * (s + u))
        if t >= tau:
            return "IIIa"
    return "IIIb"


def kurtosis_cert(t, kappa) -> dict | None:
    """Exact certificate for the kurtosis cell in regimes I, II, IIIa."""
    t, k = sp.nsimplify(t), sp.nsimplify(kappa)
    reg = kurtosis_regime(t, k)
    cons = _cons([(sp.Integer(1), "==", 1), (X, "==", 0),
                  (X ** 2, "==", 1), (X ** 4, "<=", k)])
    base = {
        "lemma_id": "kurtosis_tail", "params": {"t": _s(t), "kappa": _s(k)},
        "objective": {"sense": "sup", "type": "probability"},
        "constraints": cons,
        "provenance": {"source": f"closed form (claims L01/L02/L03, regime {reg})",
                       "date": "2026-07-02"},
        "status": "SYMBOLIC-UNVERIFIED",
    }

    if reg == "I":
        D = (t + 1 / t) ** 2
        V = 1 / (1 + t ** 2)
        base.update({
            "claim": f"P(x >= {t}) <= {_s(V)} over C({_s(k)}) [Regime I]",
            "dual_multipliers": [_s(1 / t ** 2 / D), _s(2 / t / D), _s(1 / D), "0"],
            "q_poly": _s((X + 1 / t) ** 2 / D), "bound_value": _s(V),
            "pieces": [
                {"role": "event", "domain": {"type": "half_line_ge", "a": _s(t)},
                 "blocks": [
                     {"multiplier": "1", "basis": [_s(X - t)], "gram": [[_s(1 / D)]]},
                     {"multiplier": _s(X - t), "basis": ["1"],
                      "gram": [[_s(2 * (t + 1 / t) / D)]]}]},
                {"role": "support", "domain": {"type": "real_line"},
                 "blocks": [{"multiplier": "1", "basis": ["1", "x"],
                             "gram": [[_s(1 / t ** 2 / D), _s(1 / t / D)],
                                      [_s(1 / t / D), _s(1 / D)]]}]},
            ],
            "extremal": {"atoms": [_s(t), _s(-1 / t)], "weights": [_s(V), _s(1 - V)]},
        })
        return base

    if reg == "II":
        p = (k - 1) / ((t ** 2 - 1) ** 2 + k - 1)
        u = (1 - p * t ** 2) / (1 - p)
        D2 = (t ** 2 - u) ** 2
        a = sp.sqrt(u)
        wp = ((1 - p) - p * t / a) / 2
        wm = ((1 - p) + p * t / a) / 2
        # event piece: (q-1)*D2 = y*r(y), y = x-t, r nonneg coeffs c1..c4
        y = sp.Symbol("y")
        r = sp.expand(((X ** 2 - u) ** 2 - D2).subs(X, y + t) / y)
        rp = sp.Poly(sp.expand(r), y)
        c = [sp.nsimplify(rp.coeff_monomial(y ** i)) for i in range(4)]  # c0..c3
        base.update({
            "claim": f"P(x >= {t}) <= {_s(p)} over C({_s(k)}) [Regime II]",
            "dual_multipliers": [_s(u ** 2 / D2), "0", _s(-2 * u / D2), _s(1 / D2)],
            "q_poly": _s((X ** 2 - u) ** 2 / D2), "bound_value": _s(p),
            "pieces": [
                {"role": "event", "domain": {"type": "half_line_ge", "a": _s(t)},
                 "blocks": [
                     {"multiplier": "1", "basis": [_s(X - t), _s((X - t) ** 2)],
                      "gram": [[_s(c[1] / D2), "0"], ["0", _s(c[3] / D2)]]},
                     {"multiplier": _s(X - t), "basis": ["1", _s(X - t)],
                      "gram": [[_s(c[0] / D2), "0"], ["0", _s(c[2] / D2)]]}]},
                {"role": "support", "domain": {"type": "real_line"},
                 "blocks": [{"multiplier": "1", "basis": ["1", "x", "x^2"],
                             "gram": [[_s(u ** 2 / D2), "0", _s(-u / D2)],
                                      ["0", "0", "0"],
                                      [_s(-u / D2), "0", _s(1 / D2)]]}]},
            ],
            "extremal": {"atoms": [_s(t), _s(a), _s(-a)],
                         "weights": [_s(p), _s(wp), _s(wm)]},
        })
        return base

    if reg == "IIIa":
        u, s = sp.sqrt(k - 1), sp.sqrt(k + 3)
        b = (s - u) / 2
        cc = (s + u) / 2
        k2 = u * (s - u)
        lam4 = 1 / (u * s ** 3)
        V = (1 + u / s) / 2
        r = (s ** 2 - 6 * s * u - 3 * u ** 2) / 4
        lam = [lam4 * (cc ** 4 + k2 * cc ** 2), lam4 * 2 * k2 * cc,
               lam4 * (k2 - 2 * cc ** 2), lam4]
        base.update({
            "claim": f"P(x >= {t}) <= {_s(V)} over C({_s(k)}) [Regime IIIa plateau]",
            "dual_multipliers": [_s(v) for v in lam],
            "q_poly": _s(lam4 * ((X ** 2 - cc ** 2) ** 2 + k2 * (X + cc) ** 2)),
            "bound_value": _s(V),
            "pieces": [
                {"role": "event", "domain": {"type": "half_line_ge", "a": _s(t)},
                 "blocks": [
                     {"multiplier": "1",
                      "basis": [_s((X - b) * (X - t)), _s(X - b)],
                      "gram": [[_s(lam4), "0"],
                               ["0", _s(lam4 * (t ** 2 + 2 * b * t + r))]]},
                     {"multiplier": _s(X - t), "basis": [_s(X - b)],
                      "gram": [[_s(lam4 * (2 * t + 2 * b))]]}]},
                {"role": "support", "domain": {"type": "real_line"},
                 "blocks": [{"multiplier": "1", "basis": ["1", "x", "x^2"],
                             "gram": [[_s(lam4 * (cc ** 4 + k2 * cc ** 2)), _s(lam4 * k2 * cc), _s(-lam4 * cc ** 2)],
                                      [_s(lam4 * k2 * cc), _s(lam4 * k2), "0"],
                                      [_s(-lam4 * cc ** 2), "0", _s(lam4)]]}]},
            ],
            "extremal": {"atoms": [_s(b), _s(-cc)], "weights": [_s(cc / s), _s(b / s)]},
        })
        return base

    return None  # corner / IIIb: no closed-form certificate

