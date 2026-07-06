# LemmaForge — The Exact Worst-Case Tail Probability under Bounded Kurtosis

Companion artifact for the paper

> **The Exact Worst-Case Tail Probability under Bounded Kurtosis**<br>
> Xiaoyu Li, Andi Han, Jiaojiao Jiang, Junbin Gao<br>
> Paper Link: TBD

The paper computes, exactly and for every parameter value, the worst-case tail probability

```
V1(t, κ) = sup { P(X ≥ t) :  E X = 0,  E X² = 1,  E X⁴ ≤ κ }
```

(the skewness left free). The answer is a four-regime map with closed forms on three
regimes, an explicit algebraic system on the fourth, matching extremal distributions,
sum-of-squares certificates for every bound, a one-sided/two-sided collapse theorem, and
an exact phase diagram of minimal SOS proof degree. Every certified constant in the paper
is re-verified by an **independent exact-arithmetic checker in a fresh process** — no
claim rests on floating-point evidence.

## Repository layout

| Path | Contents |
|---|---|
| `forge/` | The pipeline: DSL (`dsl.py`), SDP compiler (`compile_sdp.py`), solver wrapper (`solve.py`), independent LP cross-check (`lp_check.py`), exact-value recognition (`recognize.py`), Peyrl–Parrilo rounding + ε-retreat (`round_exact.py`), symbolic certificate constructors (`symbolic_certs.py`), and the **independent verifier** (`verify.py`, ~370 lines, shares no code with the producers) |
| `scripts/` | Reproduction drivers: symbolic identity battery, full production run, benchmark battery |
| `tests/` | Verifier known-answer tests + mutation tests (corrupted certificates must be rejected) |
| `results/` | `constants.csv` (all 131 instances with verdicts), `certificates/*.cert.json` (exact certificates, machine-readable), proof-degree map, extremal-atom data |

## Quickstart

```bash
pip install numpy scipy sympy mpmath cvxpy clarabel pyyaml pytest matplotlib

# 30-second sanity check: re-verify one shipped certificate in a fresh process
python -m forge.verify results/certificates/kurtosis_tail__kappa3_t2.cert.json
# expected: PASS VERIFIED-TIGHT: P(x >= 2) <= 2/11 over C(3) [Regime II]
```

## Full reproduction

```bash
python -m pytest tests/ -q                  # verifier known-answer + mutation tests
python scripts/verify_theorem_symbolic.py   # all 52 symbolic identities of the theorems
python scripts/produce_results.py           # full grid: solve, cross-check, round, verify
python scripts/run_b5.py                    # Khintchine p=4 battery + autonomous recognition
python scripts/verify_appendix_numbers.py   # every number printed in the worked-instances appendix
```

The full production run (131 instances) completes in tens of minutes on a single desktop
core; everything else takes seconds.

## Trust model (what you have to believe, and what you don't)

- **Not trusted:** the SDP/LP solvers, the rounding heuristics, the AI-guided search that
  proposed the conjectures. They only *discover*; nothing they output is used unverified.
- **Trusted base:** sympy's exact rational/algebraic arithmetic plus the ~370 audited
  lines of `forge/verify.py`, which re-parses each certificate, re-derives the polynomial
  identities coefficient-by-coefficient, and checks every Gram matrix PSD by two
  independent criteria (exact LDLᵀ *and* a characteristic-polynomial test) — in a fresh
  process. The verifier is mutation-tested: eight classes of corrupted certificates are
  all rejected.
- Every theorem in the paper additionally carries a hand-checkable proof (Appendix D);
  the machine layer is redundant assurance for the closed forms, and load-bearing only
  for the per-instance certification verdicts and the algebraic-degree (no-closed-form)
  statement of the central regime (Appendix C.3 of the paper states the exact scope).

## Provenance

The regime boundaries, extremal configurations, and certificate shapes were discovered by
an AI-guided search organized around this pipeline; all claims were subsequently proved
by hand-checkable arguments and re-verified in exact arithmetic. The one failed
certification is kept in the tables and dissected in the paper (Appendix G.7): a failed
certification is data, not something to hide.

## Citation

```bibtex
@misc{lemmaforge2026,
  author = {Li, Xiaoyu and Han, Andi and Jiang, Jiaojiao and Gao, Junbin},
  title  = {The Exact Worst-Case Tail Probability under Bounded Kurtosis},
  year   = {2026},
  note   = {arXiv preprint; identifier to appear},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
