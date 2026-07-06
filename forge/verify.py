"""forge.verify -- independent exact-arithmetic certificate verifier.

EPISTEMIC CONTRACT (PD1/PD2 of PLAN.md):
  * This module is the ONLY thing that grants the status CERTIFIED, and only
    when run in a fresh process (CLI: `python -m forge.verify cert.json`).
  * Dependencies: sympy + stdlib ONLY. It must never import solver code,
    compiler code, or anything else from forge.
  * It re-derives everything it checks from the certificate file itself.

What a certificate claims (univariate generalized moment problem):
    VAL = sup (or inf) of  phi(mu)  over probability measures mu on K
          subject to  E_mu[g_i(x)]  op_i  c_i   (op in {==, <=, >=})
  where phi is either P(x in S) with S = [t, inf) (type "probability")
  or E[f(x)] (type "expectation").

A certificate consists of dual multipliers lambda_i (one per moment
constraint) defining q(x) = sum_i lambda_i g_i(x), together with
sum-of-squares block decompositions witnessing the pointwise inequalities
that make q a majorant (sup) / minorant (inf) of the objective integrand,
piece by piece on canonical domains. Weak duality then gives
    sup VAL <= bound := sum_i lambda_i c_i   (resp. inf VAL >= bound).

An optional discrete "extremal" witness (atoms, weights) is checked for
exact feasibility and objective value == bound, upgrading the verdict to
VERIFIED-TIGHT (both directions, i.e. VAL == bound exactly).

Verification steps (all exact, no floats anywhere):
  V1  parse all data through sympy with Rational/exact semantics;
  V2  recompute q from (lambda, g_i); if the file carries q_poly, cross-check;
  V3  check the piece list matches the objective type/sense (majorant logic);
  V4  for each piece: every block multiplier is an ALLOWED generator of the
      declared domain, and the polynomial identity
          target(x)  ==  sum_j mult_j(x) * (z_j(x)^T G_j z_j(x))
      holds coefficient-by-coefficient over QQ;
  V5  each Gram matrix G_j is PSD, proven by exact LDL^T (with the
      zero-pivot => zero-row rule) OR by the characteristic-polynomial
      criterion det(G + s I) having all coefficients >= 0; at least one
      of the two independent checks must pass;
  V6  multiplier sign conditions for inequality-type moment constraints;
  V7  bound_value == sum_i lambda_i c_i exactly;
  V8  optional extremal witness: weights >= 0 summing to 1, all moment
      constraints satisfied exactly, objective value == bound exactly.

Exit code 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import sympy as sp

X = sp.Symbol("x", real=True)


# --------------------------------------------------------------------------
# exact parsing helpers
# --------------------------------------------------------------------------

def _num(s: Any) -> sp.Expr:
    """Parse an exact number (rational string like '3/4' or exact sympy
    expression like 'sqrt(2/3)'). Floats are rejected: certificates must be
    exact objects."""
    if isinstance(s, int):
        return sp.Integer(s)
    if isinstance(s, float):
        raise ValueError(f"float {s!r} in certificate; exact values required")
    e = sp.sympify(str(s), rational=True)
    if e.has(sp.Float):
        raise ValueError(f"non-exact value {s!r} in certificate")
    return sp.nsimplify(e, rational=False)


def _poly(s: Any) -> sp.Expr:
    """Parse a polynomial in x with exact coefficients."""
    e = sp.sympify(str(s), locals={"x": X}, rational=True)
    if e.has(sp.Float):
        raise ValueError(f"non-exact polynomial {s!r}")
    p = sp.expand(e)
    if not p.free_symbols <= {X}:
        raise ValueError(f"polynomial {s!r} has stray symbols {p.free_symbols}")
    return p


# --------------------------------------------------------------------------
# canonical domains and their allowed cone generators
# --------------------------------------------------------------------------

def domain_generators(dom: dict) -> list[sp.Expr]:
    """Return the allowed multiplier generators for a canonical domain.
    A block decomposition sum_j mult_j * SOS_j with mult_j drawn from this
    list certifies nonnegativity of the target on the domain (each generator
    is nonnegative there; this is the sound direction of Markov-Lukacs and
    is all the verifier needs)."""
    t = dom.get("type")
    if t == "real_line":
        return [sp.Integer(1)]
    if t == "half_line_ge":            # [a, +inf)
        a = _num(dom["a"])
        return [sp.Integer(1), X - a]
    if t == "half_line_le":            # (-inf, a]
        a = _num(dom["a"])
        return [sp.Integer(1), a - X]
    if t == "interval":                # [a, b]
        a, b = _num(dom["a"]), _num(dom["b"])
        if not sp.simplify(b - a).is_positive:
            raise ValueError(f"interval domain with b <= a: {dom}")
        return [sp.Integer(1), X - a, b - X, sp.expand((X - a) * (b - X))]
    raise ValueError(f"unknown domain type {t!r}")


def domain_contains(dom: dict, pt: sp.Expr) -> bool:
    """Exact membership test of a point in a canonical domain."""
    t = dom.get("type")
    if t == "real_line":
        return True
    if t == "half_line_ge":
        return bool(sp.simplify(pt - _num(dom["a"])).is_nonnegative)
    if t == "half_line_le":
        return bool(sp.simplify(_num(dom["a"]) - pt).is_nonnegative)
    if t == "interval":
        a, b = _num(dom["a"]), _num(dom["b"])
        return bool(sp.simplify(pt - a).is_nonnegative
                    and sp.simplify(b - pt).is_nonnegative)
    raise ValueError(f"unknown domain type {t!r}")


# --------------------------------------------------------------------------
# exact PSD checks (two independent methods; require at least one to pass)
# --------------------------------------------------------------------------

def psd_by_ldlt(G: sp.Matrix) -> tuple[bool, str]:
    """Exact LDL^T without pivoting: PSD iff the elimination completes with
    all pivots >= 0, where a zero pivot forces its entire row/column to be
    zero (else not PSD)."""
    n = G.shape[0]
    A = sp.Matrix(G)  # working copy
    pivots = []
    for k in range(n):
        piv = sp.nsimplify(A[k, k])
        if piv.is_negative:
            return False, f"negative pivot at {k}: {piv}"
        if piv.is_zero:
            for j in range(k, n):
                if not A[k, j].is_zero or not A[j, k].is_zero:
                    return False, f"zero pivot at {k} with nonzero row/col entry"
            pivots.append(sp.Integer(0))
            continue
        if not piv.is_positive:
            return False, f"pivot sign undecidable at {k}: {piv}"
        pivots.append(piv)
        for i in range(k + 1, n):
            f = A[i, k] / piv
            for j in range(k, n):
                A[i, j] = sp.nsimplify(sp.together(A[i, j] - f * A[k, j]))
        for i in range(k + 1, n):
            A[k, i] = sp.Integer(0)
            A[i, k] = sp.Integer(0)
    return True, f"LDLT pivots {pivots}"


def psd_by_charpoly(G: sp.Matrix) -> tuple[bool, str]:
    """G PSD iff det(G + s I) has all coefficients >= 0 (real spectrum:
    a polynomial prod (lambda_i + s) with a negative lambda_j would have a
    positive root, impossible when every coefficient is nonnegative)."""
    s = sp.Symbol("s")
    p = sp.Poly(sp.expand((G + s * sp.eye(G.shape[0])).det(method="berkowitz")), s)
    coeffs = p.all_coeffs()
    for c in coeffs:
        cc = sp.nsimplify(c)
        if cc.is_negative:
            return False, f"charpoly coefficient negative: {cc}"
        if not cc.is_nonnegative:
            return False, f"charpoly coefficient sign undecidable: {cc}"
    return True, "charpoly coefficients all >= 0"


def check_psd(G: sp.Matrix) -> tuple[bool, str]:
    if G.shape[0] != G.shape[1]:
        return False, "Gram matrix not square"
    if not G.equals(G.T):
        return False, "Gram matrix not symmetric"
    ok1, msg1 = psd_by_ldlt(G)
    ok2, msg2 = psd_by_charpoly(G)
    if ok1 or ok2:
        return True, f"ldlt: {msg1} | charpoly: {msg2}"
    return False, f"BOTH FAILED ldlt: {msg1} | charpoly: {msg2}"


# --------------------------------------------------------------------------
# core verification
# --------------------------------------------------------------------------

REQUIRED_PIECES = {
    # objective type, sense  ->  list of (piece_role, target as function of q, f)
    ("probability", "sup"): [("event", lambda q, f: q - 1), ("support", lambda q, f: q)],
    ("probability", "inf"): [("event", lambda q, f: 1 - q), ("offevent", lambda q, f: -q)],
    ("expectation", "sup"): [("support", lambda q, f: q - f)],
    ("expectation", "inf"): [("support", lambda q, f: f - q)],
}


def verify(cert: dict) -> dict:
    """Run all checks; return a report dict with 'pass' bool and details.
    Any malformed/inexact data is a verification FAILURE, never an escape."""
    report: dict[str, Any] = {"lemma_id": cert.get("lemma_id"), "checks": [], "pass": False}
    try:
        return _verify_inner(cert, report)
    except Exception as e:
        report["checks"].append({"step": "parse", "ok": False,
                                 "msg": f"{type(e).__name__}: {e}"})
        report["pass"] = False
        return report


def _verify_inner(cert: dict, report: dict) -> dict:

    def fail(step: str, msg: str) -> dict:
        report["checks"].append({"step": step, "ok": False, "msg": msg})
        return report

    def ok(step: str, msg: str = "") -> None:
        report["checks"].append({"step": step, "ok": True, "msg": msg})

    # ---- V1: objective & constraint data
    obj = cert["objective"]
    sense = obj["sense"]
    otype = obj["type"]
    if sense not in ("sup", "inf") or otype not in ("probability", "expectation"):
        return fail("V1", f"bad objective {obj}")
    f_poly = _poly(obj["poly"]) if otype == "expectation" else None
    ok("V1-objective", f"{sense} of {otype}")

    cons = cert["constraints"]
    g = [_poly(c["poly"]) for c in cons]
    ops = [c["op"] for c in cons]
    cvals = [_num(c["value"]) for c in cons]
    lam = [_num(v) for v in cert["dual_multipliers"]]
    if not (len(g) == len(ops) == len(cvals) == len(lam)):
        return fail("V1", "constraint/multiplier length mismatch")
    if not any(sp.expand(gi - 1) == 0 and op == "==" and sp.simplify(ci - 1) == 0
               for gi, op, ci in zip(g, ops, cvals)):
        return fail("V1", "missing normalization constraint E[1] == 1")
    ok("V1-constraints", f"{len(g)} moment constraints incl. normalization")

    # ---- V2: recompute q
    q = sp.expand(sum(l * gi for l, gi in zip(lam, g)))
    if "q_poly" in cert:
        if sp.expand(q - _poly(cert["q_poly"])) != 0:
            return fail("V2", "q_poly in file does not match sum lambda_i g_i")
    ok("V2-q", f"q = {sp.sstr(q)}")

    # ---- V3: piece roles match objective logic
    pieces = cert["pieces"]
    want = REQUIRED_PIECES[(otype, sense)]
    have_roles = [p["role"] for p in pieces]
    if sorted(have_roles) != sorted(r for r, _ in want):
        return fail("V3", f"piece roles {have_roles} != required {[r for r, _ in want]}")
    ok("V3-pieces", f"roles {have_roles}")

    # ---- V4/V5: each piece: identity + PSD
    for p in pieces:
        role = p["role"]
        target_fn = dict(want)[role]
        target = sp.expand(target_fn(q, f_poly))
        gens = domain_generators(p["domain"])
        total = sp.Integer(0)
        for bi, blk in enumerate(p["blocks"]):
            mult = _poly(blk["multiplier"])
            if not any(sp.expand(mult - gexp) == 0 for gexp in gens):
                return fail("V4", f"piece {role} block {bi}: multiplier {mult} "
                                  f"not an allowed generator of {p['domain']}")
            basis = [_poly(b) for b in blk["basis"]]
            n = len(basis)
            Graw = blk["gram"]
            if len(Graw) != n or any(len(r) != n for r in Graw):
                return fail("V4", f"piece {role} block {bi}: gram shape mismatch")
            G = sp.Matrix([[_num(e) for e in row] for row in Graw])
            okpsd, msg = check_psd(G)
            if not okpsd:
                return fail("V5", f"piece {role} block {bi}: Gram not PSD: {msg}")
            ok(f"V5-psd[{role}#{bi}]", msg)
            zz = sp.Matrix(basis)
            total += mult * sp.expand((zz.T * G * zz)[0, 0])
        if sp.expand(target - total) != 0:
            return fail("V4", f"piece {role}: polynomial identity FAILS; "
                              f"target - blocks = {sp.expand(target - total)}")
        ok(f"V4-identity[{role}]", "exact")

    # ---- V6: multiplier signs
    for i, (op, l) in enumerate(zip(ops, lam)):
        if op == "==":
            continue
        # sup:  E g_i <= c_i needs lambda_i >= 0 ;  E g_i >= c_i needs lambda_i <= 0
        # inf:  mirrored.
        need_nonneg = (op == "<=") if sense == "sup" else (op == ">=")
        val_ok = l.is_nonnegative if need_nonneg else l.is_nonpositive
        if not val_ok:
            return fail("V6", f"constraint {i} op {op}: multiplier {l} has wrong sign")
    ok("V6-signs")

    # ---- V7: bound value
    bound = _num(cert["bound_value"])
    implied = sp.simplify(sum(l * c for l, c in zip(lam, cvals)))
    if sp.simplify(implied - bound) != 0:
        return fail("V7", f"bound_value {bound} != sum lambda_i c_i = {implied}")
    ok("V7-bound", f"bound = {bound} ({'upper' if sense == 'sup' else 'lower'})")

    # ---- V8: optional extremal witness (tightness)
    tight = False
    if "extremal" in cert and cert["extremal"]:
        ext = cert["extremal"]
        atoms = [_num(a) for a in ext["atoms"]]
        wts = [_num(w) for w in ext["weights"]]
        if len(atoms) != len(wts):
            return fail("V8", "atoms/weights length mismatch")
        for w in wts:
            if not w.is_nonnegative:
                return fail("V8", f"negative weight {w}")
        if sp.simplify(sum(wts) - 1) != 0:
            return fail("V8", "weights do not sum to 1")
        for gi, op, ci in zip(g, ops, cvals):
            m = sp.simplify(sum(w * gi.subs(X, a) for w, a in zip(wts, atoms)))
            d = sp.simplify(m - ci)
            cond = (d == 0) if op == "==" else (d.is_nonpositive if op == "<=" else d.is_nonnegative)
            if not cond:
                return fail("V8", f"witness violates E[{sp.sstr(gi)}] {op} {ci}: got {m}")
        # objective value of the witness
        if otype == "probability":
            edom = next(p["domain"] for p in pieces if p["role"] == "event")
            val = sp.simplify(sum(w for w, a in zip(wts, atoms) if domain_contains(edom, a)))
        else:
            val = sp.simplify(sum(w * f_poly.subs(X, a) for w, a in zip(wts, atoms)))
        if sp.simplify(val - bound) != 0:
            return fail("V8", f"witness objective {val} != bound {bound} (not tight)")
        tight = True
        ok("V8-extremal", f"feasible witness attains bound exactly ({len(atoms)} atoms)")

    report["pass"] = True
    report["verdict"] = "VERIFIED-TIGHT" if tight else "VERIFIED-BOUND"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LemmaForge independent certificate verifier")
    ap.add_argument("cert", help="path to certificate JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    with open(args.cert, "r", encoding="utf-8") as fh:
        cert = json.load(fh)
    try:
        report = verify(cert)
    except Exception as e:  # any parse/shape error is a verification failure
        print(f"VERIFY ERROR: {type(e).__name__}: {e}")
        return 1
    if not args.quiet:
        for c in report["checks"]:
            print(f"  [{'ok' if c['ok'] else 'XX'}] {c['step']}: {c.get('msg','')}")
    if report["pass"]:
        print(f"PASS {report['verdict']}: {cert.get('claim', cert.get('lemma_id'))}"
              f"  bound={cert.get('bound_value')}")
        return 0
    print(f"FAIL: {report['checks'][-1]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
