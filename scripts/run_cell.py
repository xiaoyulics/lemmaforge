"""Quick driver: run one cell across its grid; SDP vs LP vs truth."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp
from forge.dsl import load_spec, instantiate, grid_points
from forge.compile_sdp import compile_dual
from forge.solve import solve
from forge.lp_check import lp_value

spec = load_spec(sys.argv[1])
for pt in grid_points(spec):
    inst = instantiate(spec, pt)
    sdp = compile_dual(inst)
    rec = solve(sdp)
    lpv = lp_value(inst)
    truth = float(inst.truth) if inst.truth is not None else float("nan")
    line = (f"{inst.label:34s} SDP={rec.get('value', float('nan')):.10f} "
            f"LP={lpv:.10f} truth={truth:.10f} "
            f"res={rec.get('max_residual', -1):.1e} status={rec['status']}")
    print(line)
    mom = rec.get("moments")
    if mom and "--moments" in sys.argv:
        for role, y in mom.items():
            print(f"    {role:8s} moments: {[round(v, 6) for v in y]}")
    if "--lam" in sys.argv and rec.get("lambda") is not None:
        print(f"    lambda: {[round(float(v), 8) for v in rec['lambda']]}")
