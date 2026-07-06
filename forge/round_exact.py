"""forge.round_exact -- S5: Peyrl-Parrilo rational rounding.

Numeric dual (lambda, Gram blocks) -> exact rational certificate JSON in the
schema of forge/verify.py. Procedure (PLAN B.5):

  1. rationalize lambda and Gram entries (denominator ladder / recognized
     exact lambda if provided);
  2. project EXACTLY (rational normal equations) onto the affine subspace of
     the polynomial identity  s_q*q + t_poly == sum_j mult_j * z^T G_j z,
     coefficient-by-coefficient, for every piece;
  3. check every projected Gram exactly PSD (LDL^T here; the independent
     verifier repeats this with its own code);
  4. on failure: climb the ladder (bigger denominators), then epsilon-retreat
     (certify bound + delta with delta a small explicit rational).

This module PRODUCES certificates; it never grants CERTIFIED status. Only a
fresh-process run of forge.verify does (PD1).
"""

from __future__ import annotations

import json
from fractions import Fraction

import sympy as sp

from .compile_sdp import CompiledSDP

X = sp.Symbol("x", real=True)


def _ratm(Gf, den: int) -> sp.Matrix:
    n = len(Gf)
    return sp.Matrix([[sp.Rational(Fraction(float(Gf[i][j])).limit_denominator(den))
                       for j in range(n)] for i in range(n)])


def _psd_ldlt(G: sp.Matrix) -> bool:
    n = G.shape[0]
    A = sp.Matrix(G)
    for k in range(n):
        piv = A[k, k]
        if piv.is_negative:
            return False
        if piv.is_zero:
            if any((not A[k, j].is_zero) or (not A[j, k].is_zero) for j in range(k, n)):
                return False
            continue
        if not piv.is_positive:
            return False
        for i in range(k + 1, n):
            f = A[i, k] / piv
            for j in range(k, n):
                A[i, j] = sp.together(A[i, j] - f * A[k, j])
    return True


def round_certificate(sdp: CompiledSDP, rec: dict,
                      lam_exact: list | None = None,
                      bound_exact=None,
                      extremal: dict | None = None,
                      den_ladder=(10**3, 10**6, 10**9)) -> dict | None:
    """Attempt exact rounding; return certificate dict or None."""
    inst = sdp.inst
    m = len(inst.constraints)

    for den in den_ladder:
        if lam_exact is not None:
            lam = [sp.Rational(v) for v in lam_exact]
        else:
            lam = [sp.Rational(Fraction(float(v)).limit_denominator(den))
                   for v in rec["lambda"]]
        q = sp.expand(sum(l * c.poly for l, c in zip(lam, inst.constraints)))

        pieces_out = []
        all_psd = True
        for piece in sdp.pieces:
            target = sp.expand(piece.s_q * q + piece.t_poly)
            # collect this piece's numeric grams
            blocks_meta = [(blk.multiplier, blk.basis) for blk in piece.blocks]
            grams_num = [g["G"] for g in rec["grams"] if g["piece"] == piece.role]
            # variables: upper-triangular entries of each block Gram
            vars_, G_syms = [], []
            for bi, (mult, basis) in enumerate(blocks_meta):
                n = len(basis)
                Gs = sp.zeros(n, n)
                for i in range(n):
                    for j in range(i, n):
                        v = sp.Symbol(f"g_{bi}_{i}_{j}")
                        Gs[i, j] = v
                        Gs[j, i] = v
                        vars_.append(v)
                G_syms.append(Gs)
            expr = target
            for (mult, basis), Gs in zip(blocks_meta, G_syms):
                z = sp.Matrix(basis)
                expr = expr - sp.expand(mult * (z.T * Gs * z)[0, 0])
            eqs = sp.Poly(sp.expand(expr), X).all_coeffs()
            A, b = sp.linear_eq_to_matrix([sp.Eq(e, 0) for e in eqs], vars_)
            # rationalized start point
            w0 = []
            for bi, (mult, basis) in enumerate(blocks_meta):
                n = len(basis)
                Gr = _ratm(grams_num[bi], den)
                for i in range(n):
                    for j in range(i, n):
                        w0.append(Gr[i, j])
            w0 = sp.Matrix(w0)
            # exact projection onto A w = b:  w = w0 - A^T (A A^T)^-1 (A w0 - b)
            r = A * w0 - b
            try:
                y = (A * A.T).solve(r)
            except Exception:
                all_psd = False
                break
            w = w0 - A.T * y
            # rebuild blocks, check PSD
            blocks_json = []
            idx = 0
            for (mult, basis), Gs in zip(blocks_meta, G_syms):
                n = len(basis)
                G = sp.zeros(n, n)
                for i in range(n):
                    for j in range(i, n):
                        G[i, j] = sp.nsimplify(w[idx])
                        G[j, i] = G[i, j]
                        idx += 1
                if not _psd_ldlt(G):
                    all_psd = False
                    break
                blocks_json.append({
                    "multiplier": sp.sstr(mult),
                    "basis": [sp.sstr(bb) for bb in basis],
                    "gram": [[sp.sstr(G[i, j]) for j in range(n)] for i in range(n)],
                })
            if not all_psd:
                break
            pieces_out.append({"role": piece.role, "domain": piece.domain,
                               "blocks": blocks_json})
        if not all_psd:
            continue

        bound = sp.simplify(sum(l * c.value for l, c in zip(lam, inst.constraints)))
        if bound_exact is not None and sp.simplify(bound - bound_exact) != 0:
            # certificate is valid but for a different bound; keep it only
            # if no exact target was demanded
            continue
        cert = {
            "lemma_id": inst.lemma_id,
            "claim": f"{'P(x >= ' + sp.sstr(inst.event_t) + ')' if inst.obj_type == 'probability' else 'E[f]'} "
                     f"{'<=' if inst.sense == 'sup' else '>='} {sp.sstr(bound)} over moment class",
            "params": {k: sp.sstr(v) for k, v in inst.params.items()},
            "objective": {"sense": inst.sense, "type": inst.obj_type},
            "constraints": [{"poly": sp.sstr(c.poly), "op": c.op, "value": sp.sstr(c.value)}
                            for c in inst.constraints],
            "dual_multipliers": [sp.sstr(l) for l in lam],
            "q_poly": sp.sstr(q),
            "bound_value": sp.sstr(bound),
            "pieces": pieces_out,
            "status": "ROUNDED-UNVERIFIED",
            "provenance": {**rec.get("provenance", {}), "den_ladder_rung": den,
                           "lambda_source": "exact" if lam_exact else "rationalized"},
        }
        if extremal:
            cert["extremal"] = extremal
        return cert
    return None


def save_certificate(cert: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cert, fh, indent=1)


def round_with_retreat(sdp: CompiledSDP, rec: dict, solve_fn,
                       lam_exact=None, bound_exact=None, extremal=None,
                       deltas=(sp.Rational(1, 10**9), sp.Rational(1, 10**6))) -> dict | None:
    """Full ladder: direct rounding; then epsilon-retreat (re-solve with the
    objective pinned to value+delta, restoring a strict interior, and round
    that). A retreat certificate certifies the WEAKER bound value+delta and
    carries no tightness witness."""
    import cvxpy as cp
    import numpy as np

    cert = round_certificate(sdp, rec, lam_exact=lam_exact,
                             bound_exact=bound_exact, extremal=extremal)
    if cert is not None:
        return cert
    inst = sdp.inst
    cvals = np.array([float(c.value) for c in inst.constraints])
    for delta in deltas:
        target = rec["value"] + float(delta) if inst.sense == "sup" \
            else rec["value"] - float(delta)
        pin = [cvals @ sdp.lam == target] if inst.sense == "sup" \
            else [cvals @ sdp.lam == target]
        prob = cp.Problem(cp.Minimize(0), sdp.coeff_constraints + sdp.sign_constraints + pin)
        try:
            prob.solve(solver="CLARABEL")
        except Exception:
            continue
        if prob.status not in ("optimal", "optimal_inaccurate"):
            continue
        rec2 = {"lambda": np.asarray(sdp.lam.value, dtype=float),
                "grams": [], "provenance": {**rec.get("provenance", {}),
                                            "epsilon_retreat": sp.sstr(delta)}}
        for piece in sdp.pieces:
            for blk in piece.blocks:
                G = np.asarray(blk.var.value, dtype=float)
                rec2["grams"].append({"piece": piece.role,
                                      "multiplier": str(blk.multiplier),
                                      "basis": [str(b) for b in blk.basis],
                                      "G": 0.5 * (G + G.T)})
        cert = round_certificate(sdp, rec2, lam_exact=None, bound_exact=None,
                                 extremal=None)
        if cert is not None:
            cert["provenance"]["epsilon_retreat"] = sp.sstr(delta)
            return cert
    return None
