"""Master production run (S1-S6 end-to-end for every cell).

Emits:
  results/constants.csv               one row per (cell, point)
  results/certificates/*.cert.json    exact certificates
  results/extremal_gallery/gallery.csv
  results/proof_degree_map.csv
  results/pseudo_counterexamples/*.pce.json (degree-2 obstructions)
Status is CERTIFIED only after `python -m forge.verify` PASSES IN A FRESH
SUBPROCESS (PD1). Everything else stays NUMERICAL/RECOGNIZED/SYMBOLIC.
"""
import csv
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import sympy as sp

from forge.dsl import load_spec, instantiate, grid_points
from forge.compile_sdp import compile_dual
from forge.solve import solve
from forge.lp_check import lp_value
from forge.round_exact import round_with_retreat, save_certificate
from forge import symbolic_certs as SC

RES = os.path.join(ROOT, "results")
CERTD = os.path.join(RES, "certificates")
os.makedirs(CERTD, exist_ok=True)
os.makedirs(os.path.join(RES, "extremal_gallery"), exist_ok=True)
os.makedirs(os.path.join(RES, "pseudo_counterexamples"), exist_ok=True)


def fresh_verify(path: str) -> bool:
    env = dict(os.environ, PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-m", "forge.verify", path, "--quiet"],
                       capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)
    return r.returncode == 0


def label(params):
    return "_".join(f"{k}{str(v).replace('/', 'over')}" for k, v in sorted(params.items()))


def symbolic_cert_for(inst):
    lid = inst.lemma_id
    P = inst.params
    if lid == "markov" and P["t"] >= 1:
        return SC.markov_cert(P["t"])
    if lid == "cantelli":
        return SC.cantelli_cert(P["sigma2"], P["t"])
    if lid == "paley_zygmund":
        return SC.paley_zygmund_cert(P["m2"], P["theta"])
    if lid == "kurtosis_tail":
        return SC.kurtosis_cert(P["t"], P["kappa"])
    return None


def exact_truth(inst):
    if inst.truth is not None:
        return sp.nsimplify(inst.truth)
    if inst.lemma_id == "kurtosis_tail":
        t, k = inst.params["t"], inst.params["kappa"]
        reg = SC.kurtosis_regime(t, k)
        t, k = sp.nsimplify(t), sp.nsimplify(k)
        if reg == "I":
            return 1 / (1 + t ** 2)
        if reg == "II":
            return (k - 1) / ((t ** 2 - 1) ** 2 + k - 1)
        if reg == "IIIa":
            return (1 + sp.sqrt((k - 1) / (k + 3))) / 2
    return None


rows, gal_rows, pd_rows = [], [], []
cells = [
    "lemmas/benchmarks/b1_markov.yaml",
    "lemmas/benchmarks/b2_cantelli.yaml",
    "lemmas/benchmarks/b4_paley_zygmund.yaml",
    "lemmas/benchmarks/b3_three_moment.yaml",
    "lemmas/frontier/f1_kurtosis.yaml",
    "lemmas/frontier/f1_skew_kurtosis.yaml",
]

for cell in cells:
    spec = load_spec(os.path.join(ROOT, cell))
    for pt in grid_points(spec):
        inst = instantiate(spec, pt)
        sdp = compile_dual(inst)
        rec = solve(sdp)
        if rec.get("status") not in ("optimal", "optimal_inaccurate"):
            rows.append(dict(lemma_id=inst.lemma_id, params=json.dumps({k: str(v) for k, v in inst.params.items()}),
                             degree=inst.degree, sense=inst.sense, value_exact="", value_float="",
                             status="SOLVER_FAILED", truth="", cert_path="", solver="", residual="", note=""))
            continue
        try:
            lpv = lp_value(inst)
        except Exception:
            lpv = float("nan")
        vfloat = rec["value"]

        # certificate: symbolic constructor first, else rounding ladder
        cert = None
        source = ""
        try:
            cert = symbolic_cert_for(inst)
            source = "symbolic" if cert else ""
        except Exception as e:
            cert = None
        if cert is None:
            try:
                cert = round_with_retreat(sdp, rec, solve_fn=solve)
                source = "rounded" if cert else ""
            except Exception:
                cert = None

        status = "NUMERICAL"
        cert_path = ""
        exact = exact_truth(inst)
        if cert is not None:
            cert_path = os.path.join(CERTD, f"{inst.lemma_id}__{label(inst.params)}.cert.json")
            save_certificate(cert, cert_path)
            ok = fresh_verify(cert_path)
            if ok:
                status = "CERTIFIED"
                cert["status"] = "CERTIFIED"
                save_certificate(cert, cert_path)
            else:
                status = "ROUNDED-FAILED-VERIFY"
        # sanity: certificate bound vs SDP value
        bound_gap = ""
        if cert is not None:
            bv = float(sp.nsimplify(cert["bound_value"]))
            bound_gap = f"{abs(bv - vfloat):.2e}"

        lp_gap = abs(vfloat - lpv) if lpv == lpv else float("nan")
        rows.append(dict(
            lemma_id=inst.lemma_id,
            params=json.dumps({k: str(v) for k, v in inst.params.items()}),
            degree=inst.degree, sense=inst.sense,
            value_exact=(sp.sstr(exact) if exact is not None else ""),
            value_float=f"{vfloat:.12g}",
            status=status,
            truth=(sp.sstr(sp.nsimplify(inst.truth)) if inst.truth is not None else ""),
            cert_path=os.path.relpath(cert_path, ROOT) if cert_path else "",
            solver=rec["provenance"]["solver"], residual=f"{rec['max_residual']:.1e}",
            note=f"lp_gap={lp_gap:.1e};src={source};certgap={bound_gap}",
        ))

        # gallery
        if cert is not None and "extremal" in cert:
            gal_rows.append(dict(lemma_id=inst.lemma_id,
                                 params=rows[-1]["params"],
                                 atoms=";".join(cert["extremal"]["atoms"]),
                                 weights=";".join(cert["extremal"]["weights"]),
                                 kind="exact"))
        elif rec.get("moments"):
            mom = {k: [round(float(v), 9) for v in y] for k, y in rec["moments"].items()}
            gal_rows.append(dict(lemma_id=inst.lemma_id, params=rows[-1]["params"],
                                 atoms=json.dumps(mom), weights="", kind="split-moments"))

        # proof-degree map for the kurtosis cell
        if inst.lemma_id == "kurtosis_tail":
            t, k = sp.nsimplify(inst.params["t"]), sp.nsimplify(inst.params["kappa"])
            kc = (t ** 4 - t ** 2 + 1) / t ** 2
            pd = 2 if k >= kc else 4
            pd_rows.append(dict(t=sp.sstr(t), kappa=sp.sstr(k),
                                regime=SC.kurtosis_regime(t, k), proof_degree=pd,
                                deg2_value=sp.sstr(1 / (1 + t ** 2)),
                                true_value=rows[-1]["value_exact"] or rows[-1]["value_float"]))
            if pd == 4:
                pce = {
                    "refuted_claim": f"'P(x>={sp.sstr(t)}) <= V1' has a degree-2 SOS certificate over C({sp.sstr(k)})",
                    "witness": "moments (1,0,1) truncated at degree 2 = Cantelli extremal (a true distribution in the 2-moment class); any degree-2-feasible q has E[q] >= 1/(1+t^2) > V1",
                    "deg2_optimal_value": sp.sstr(1 / (1 + t ** 2)),
                    "true_value_deg4": rows[-1]["value_exact"] or rows[-1]["value_float"],
                    "moments": ["1", "0", "1"],
                    "provenance": rec["provenance"],
                }
                with open(os.path.join(RES, "pseudo_counterexamples",
                                       f"deg2__{label(inst.params)}.pce.json"), "w", encoding="utf-8") as fh:
                    json.dump(pce, fh, indent=1)

        print(f"{inst.label:44s} {status:22s} v={vfloat:.9f} lpgap={lp_gap:.1e} src={source}")

with open(os.path.join(RES, "constants.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(RES, "extremal_gallery", "gallery.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(gal_rows[0].keys()))
    w.writeheader()
    w.writerows(gal_rows)
with open(os.path.join(RES, "proof_degree_map.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(pd_rows[0].keys()))
    w.writeheader()
    w.writerows(pd_rows)

n_cert = sum(1 for r in rows if r["status"] == "CERTIFIED")
print(f"\nDONE: {len(rows)} points, {n_cert} CERTIFIED, "
      f"{sum(1 for r in rows if r['status'] == 'NUMERICAL')} numerical, "
      f"{sum(1 for r in rows if 'FAIL' in r['status'])} failed-verify")
