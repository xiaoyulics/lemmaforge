"""B5 -- Khintchine p=4 (POP mode validation).

maximize  E(sum a_i eps_i)^4 = sum_i a_i^4 + 6 sum_{i<j} a_i^2 a_j^2
subject to sum a_i^2 = 1,  n = 2..8.
Truth: 3 - 2/n (attained at a_i = 1/sqrt(n); Haagerup 1981 for the constant
as n->oo). Level-2 Lasserre / degree-4 SOS-on-sphere is exact here via the
identity n*sum a^4 - (sum a^2)^2 = sum_{i<j} (a_i^2 - a_j^2)^2.

Validation gates (PLAN Phase 1):
  (i) SDP values match 3 - 2/n to 1e-7 for n = 2..8;
 (ii) forge.recognize.fit_sequence outputs exactly 3 - 2/n from the floats,
      autonomously (no hints).
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cvxpy as cp
import sympy as sp

from forge.recognize import fit_sequence


def solve_khintchine_pop(n: int) -> float:
    # monomial index maps
    def monos_up_to(d):
        out = []
        for total in range(d + 1):
            for combo in itertools.combinations_with_replacement(range(n), total):
                e = [0] * n
                for i in combo:
                    e[i] += 1
                out.append(tuple(e))
        return out

    basis2 = monos_up_to(2)                    # SOS basis
    all4 = {m: idx for idx, m in enumerate(monos_up_to(4))}
    N = len(basis2)

    G = cp.Variable((N, N), PSD=True)
    mu = cp.Variable(N)                        # multiplier coeffs on basis2 monomials
    gam = cp.Variable()

    # target coefficients: gamma - f
    fcoef = np.zeros(len(all4))
    for i in range(n):
        e = [0] * n; e[i] = 4
        fcoef[all4[tuple(e)]] += 1.0
    for i, j in itertools.combinations(range(n), 2):
        e = [0] * n; e[i] = 2; e[j] = 2
        fcoef[all4[tuple(e)]] += 6.0

    # build equality per degree<=4 monomial:
    #   gamma*[m==1] - f_m  ==  <G-contributions> + <mu*(sum a^2 - 1)>_m
    lhs_terms = [[] for _ in range(len(all4))]
    for p in range(N):
        for q in range(p, N):
            m = tuple(a + b for a, b in zip(basis2[p], basis2[q]))
            coef = 1.0 if p == q else 2.0
            lhs_terms[all4[m]].append(coef * G[p, q])
    for k, m in enumerate(basis2):
        # +mu_k * a_i^2 * m  and  -mu_k * m
        lhs_terms[all4[m]].append(-1.0 * mu[k])
        for i in range(n):
            e = list(m); e[i] += 2
            lhs_terms[all4[tuple(e)]].append(1.0 * mu[k])

    cons = []
    zero = tuple([0] * n)
    for m, idx in all4.items():
        lhs = cp.sum(cp.hstack(lhs_terms[idx])) if lhs_terms[idx] else 0
        rhs = (gam if m == zero else 0) - fcoef[idx]
        cons.append(lhs == rhs)

    prob = cp.Problem(cp.Minimize(gam), cons)
    prob.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_gap_rel=1e-11)
    return float(prob.value)


if __name__ == "__main__":
    ns, vals = [], []
    for n in range(2, 9):
        v = solve_khintchine_pop(n)
        truth = 3 - 2 / n
        ns.append(n); vals.append(v)
        print(f"n={n}: SDP={v:.10f} truth={truth:.10f} |diff|={abs(v-truth):.1e}")
    fit = fit_sequence(ns, vals)
    print(f"\nrecognize.fit_sequence -> {fit['expr'] if fit else 'FAILED'}"
          f"  (basis {fit['basis'] if fit else '-'}, residual {fit['max_residual'] if fit else '-'})")
    ok = fit and sp.simplify(fit["expr"] - (3 - 2 / sp.Symbol("n"))) == 0
    print("B5 VALIDATION:", "PASS" if ok and all(abs(v - (3 - 2/n)) < 1e-7 for n, v in zip(ns, vals)) else "FAIL")
