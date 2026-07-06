"""forge.compile_sdp -- univariate exact SOS compiler (S2).

Compiles a univariate MOMENT-mode LemmaInstance into the dual SOS
semidefinite program, using the exact Markov-Lukacs representations
(PLAN section 2.4) -- never a generic Putinar template. The primal
pseudo-moments are recovered as the duals of the coefficient-matching
equalities, so one solve yields both the bound and the extremal moments.

Piece logic (must mirror forge/verify.py exactly -- the verifier is the
authority; this module just searches inside the verifier's certificate
language):
    P-sup : (event,   K cap S, target q - 1), (support, K, target q)
    P-inf : (event,   K cap S, target 1 - q), (offevent, cl(K minus S), target -q)
    E-sup : (support, K, target q - f)
    E-inf : (support, K, target f - q)
"""

from __future__ import annotations

import dataclasses

import cvxpy as cp
import numpy as np
import sympy as sp

from .dsl import LemmaInstance

X = sp.Symbol("x", real=True)


@dataclasses.dataclass
class Block:
    multiplier: sp.Expr           # allowed generator of the domain
    basis: list[sp.Expr]          # polynomial basis entries (shifted)
    var: cp.Variable              # PSD Gram variable


@dataclasses.dataclass
class Piece:
    role: str
    domain: dict                  # canonical domain dict (verifier schema)
    s_q: int                      # target = s_q * q + t_poly
    t_poly: sp.Expr
    blocks: list[Block]
    coeff_constraints: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CompiledSDP:
    inst: LemmaInstance
    lam: cp.Variable
    pieces: list[Piece]
    coeff_constraints: list       # cvxpy equality per monomial power 0..D
    sign_constraints: list
    problem: cp.Problem
    D: int


def _domain_blocks(domain: dict, D: int) -> list[tuple[sp.Expr, list[sp.Expr]]]:
    """Markov-Lukacs block structure: list of (multiplier, basis polys)."""
    t = domain["type"]
    if t == "real_line":
        if D % 2 != 0:
            raise ValueError("odd-degree target cannot be SOS on R")
        m = D // 2
        return [(sp.Integer(1), [X**k for k in range(m + 1)])]
    if t in ("half_line_ge", "half_line_le"):
        a = sp.Rational(domain["a"])
        y = (X - a) if t == "half_line_ge" else (a - X)
        m0, m1 = D // 2, (D - 1) // 2
        out = [(sp.Integer(1), [y**k for k in range(m0 + 1)])]
        if m1 >= 0:
            out.append((sp.expand(y), [y**k for k in range(m1 + 1)]))
        return out
    if t == "interval":
        a, b = sp.Rational(domain["a"]), sp.Rational(domain["b"])
        w = sp.expand((X - a) * (b - X))
        if D % 2 == 0:
            m = D // 2
            out = [(sp.Integer(1), [(X - a)**k for k in range(m + 1)])]
            if m - 1 >= 0:
                out.append((w, [(X - a)**k for k in range(m)]))
            return out
        m = (D - 1) // 2
        return [(sp.expand(X - a), [(X - a)**k for k in range(m + 1)]),
                (sp.expand(b - X), [(X - a)**k for k in range(m + 1)])]
    raise ValueError(f"domain {t}")


def _pieces_for(inst: LemmaInstance) -> list[tuple[str, dict, int, sp.Expr]]:
    """(role, domain, s_q, t_poly) per the piece-logic table."""
    K_dom = ({"type": "real_line"} if inst.support == "R"
             else {"type": "half_line_ge", "a": str(inst.support_a)})
    if inst.obj_type == "expectation":
        f = inst.obj_poly
        if inst.sense == "sup":
            return [("support", K_dom, +1, sp.expand(-f))]
        return [("support", K_dom, -1, sp.expand(+f))]

    t = inst.event_t
    if inst.support == "R":
        S_dom = {"type": "half_line_ge", "a": str(t)}
        comp_dom = {"type": "half_line_le", "a": str(t)}
    else:
        a = inst.support_a
        if not (t >= a):
            raise ValueError("event threshold below support start")
        S_dom = {"type": "half_line_ge", "a": str(t)}
        comp_dom = {"type": "interval", "a": str(a), "b": str(t)}
    if inst.sense == "sup":
        return [("event", S_dom, +1, sp.Integer(-1)), ("support", K_dom, +1, sp.Integer(0))]
    return [("event", S_dom, -1, sp.Integer(1)), ("offevent", comp_dom, -1, sp.Integer(0))]


def compile_dual(inst: LemmaInstance) -> CompiledSDP:
    if inst.mode != "MOMENT":
        raise ValueError("compile_dual handles MOMENT mode univariate cells")
    D = inst.degree
    m = len(inst.constraints)
    lam = cp.Variable(m, name="lambda")

    g_coeffs = np.zeros((m, D + 1))
    for i, c in enumerate(inst.constraints):
        p = sp.Poly(c.poly, X)
        if p.degree() > D:
            raise ValueError("constraint degree exceeds cell degree")
        for k, coef in enumerate(reversed(p.all_coeffs())):
            g_coeffs[i, k] = float(coef)

    pieces: list[Piece] = []
    coeff_cons = []
    for role, domain, s_q, t_poly in _pieces_for(inst):
        blocks = []
        for mult, basis in _domain_blocks(domain, D):
            n = len(basis)
            G = cp.Variable((n, n), PSD=True, name=f"G_{role}_{sp.sstr(mult)[:8]}")
            blocks.append(Block(mult, basis, G))
        pieces.append(Piece(role, domain, s_q, sp.expand(t_poly), blocks))

    # coefficient matching per piece, per power k = 0..D
    for piece in pieces:
        tp = sp.Poly(piece.t_poly, X)
        t_vec = np.zeros(D + 1)
        for k, coef in enumerate(reversed(tp.all_coeffs())):
            t_vec[k] = float(coef)
        # rhs: sum over blocks of <M_k, G>
        rhs_terms_per_k: list[list] = [[] for _ in range(D + 1)]
        for blk in piece.blocks:
            n = len(blk.basis)
            for u in range(n):
                for v in range(n):
                    prod = sp.Poly(sp.expand(blk.multiplier * blk.basis[u] * blk.basis[v]), X)
                    if prod.degree() > D:
                        raise ValueError("block product exceeds degree budget")
                    for k, coef in enumerate(reversed(prod.all_coeffs())):
                        if coef != 0:
                            rhs_terms_per_k[k].append(float(coef) * blk.var[u, v])
        piece_cons = []
        for k in range(D + 1):
            lhs = piece.s_q * (g_coeffs[:, k] @ lam) + t_vec[k]
            rhs = cp.sum(cp.hstack(rhs_terms_per_k[k])) if rhs_terms_per_k[k] else 0
            piece_cons.append(lhs == rhs)
        coeff_cons.extend(piece_cons)
        piece.coeff_constraints = piece_cons

    # multiplier sign conditions
    sign_cons = []
    for i, c in enumerate(inst.constraints):
        if c.op == "==":
            continue
        need_nonneg = (c.op == "<=") if inst.sense == "sup" else (c.op == ">=")
        sign_cons.append(lam[i] >= 0 if need_nonneg else lam[i] <= 0)

    cvals = np.array([float(c.value) for c in inst.constraints])
    objective = cp.Minimize(cvals @ lam) if inst.sense == "sup" else cp.Maximize(cvals @ lam)
    prob = cp.Problem(objective, coeff_cons + sign_cons)
    return CompiledSDP(inst, lam, pieces, coeff_cons, sign_cons, prob, D)


def pseudo_moments(sdp: CompiledSDP) -> dict[str, np.ndarray] | None:
    """Extract the primal (pseudo-)moment sequences of the SPLIT measures:
    the duals of each piece's coefficient equalities are the moments of that
    piece's measure (event piece -> nu on K cap S with mass = tail prob;
    support piece -> rho on K; single piece for E-type -> mu itself).
    Sign convention is resolved by requiring total mass (sum over pieces of
    y[0]) to be +1; cvxpy negates duals between Minimize/Maximize."""
    out: dict[str, np.ndarray] = {}
    for piece in sdp.pieces:
        duals = [c.dual_value for c in piece.coeff_constraints]
        if any(d is None for d in duals):
            return None
        out[piece.role] = np.array([float(d) for d in duals])
    total_mass = sum(y[0] for y in out.values())
    if abs(total_mass) < 1e-9:
        return None
    if total_mass < 0:
        out = {k: -v for k, v in out.items()}
    return out
