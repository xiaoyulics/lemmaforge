"""forge.dsl -- lemma specs (S1).

YAML schema (PLAN.md section 5 S1), univariate MOMENT mode and POP mode.
All numbers are parsed through sympy.Rational; floats in specs are rejected
unless exactly representable (they are normalized through Rational anyway).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import sympy as sp
import yaml

X = sp.Symbol("x", real=True)


def _rat(v: Any) -> sp.Rational:
    if isinstance(v, float):
        r = sp.Rational(str(v))
    else:
        r = sp.Rational(v)
    return r


def _poly(s: str, params: dict[str, sp.Rational]) -> sp.Expr:
    e = sp.sympify(str(s), locals={"x": X, **params}, rational=True)
    if e.has(sp.Float):
        raise ValueError(f"non-exact polynomial {s!r}")
    return sp.expand(e)


@dataclasses.dataclass
class Constraint:
    poly: sp.Expr
    op: str          # '==', '<=', '>='
    value: sp.Rational


@dataclasses.dataclass
class LemmaInstance:
    """A lemma spec with all parameters substituted to exact rationals."""
    lemma_id: str
    mode: str                     # MOMENT | POP
    support: str                  # 'R' | 'half_line[a]' | 'interval[a,b]'  (univariate MOMENT)
    support_a: sp.Rational | None
    support_b: sp.Rational | None
    constraints: list[Constraint]
    sense: str                    # sup | inf
    obj_type: str                 # probability | expectation
    event_t: sp.Rational | None   # event = [t, inf)   (probability type)
    obj_poly: sp.Expr | None      # expectation type
    degree: int                   # top polynomial degree D (certificate degree)
    params: dict[str, sp.Rational]
    truth: sp.Expr | None
    tags: list[str]

    @property
    def label(self) -> str:
        ps = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.lemma_id}[{ps}]" if ps else self.lemma_id


def load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    for key in ("id", "mode", "objective"):
        if key not in spec:
            raise ValueError(f"{path}: missing required key {key!r}")
    return spec


def instantiate(spec: dict, point: dict[str, Any] | None = None) -> LemmaInstance:
    """Build a LemmaInstance at a parameter point (defaults if omitted)."""
    pdefs = spec.get("params", {}) or {}
    params: dict[str, sp.Rational] = {}
    for name, meta in pdefs.items():
        raw = (point or {}).get(name, meta.get("default"))
        if raw is None:
            raise ValueError(f"param {name} has no value")
        params[name] = _rat(raw)
    psyms = {k: sp.Rational(v) for k, v in params.items()}

    cons = []
    for c in spec.get("moments", []) or []:
        cons.append(Constraint(_poly(c["poly"], psyms), c["op"], sp.Rational(sp.sympify(str(c["value"]), locals=psyms))))

    obj = spec["objective"]
    sense = obj["sense"]
    otype = obj["type"]
    event_t = None
    obj_poly = None
    if otype == "probability":
        # canonical event: x - t >= 0
        ev = obj["event"]
        if isinstance(ev, list):
            ev = ev[0]
        lhs = _poly(str(ev).split(">=")[0], psyms)
        d = sp.degree(lhs, X)
        if d != 1 or sp.LC(lhs, X) != 1:
            raise ValueError("event must be of the form 'x - t >= 0'")
        event_t = sp.Rational(-lhs.subs(X, 0))
    elif otype == "expectation":
        obj_poly = _poly(obj["poly"], psyms)
    else:
        raise ValueError(f"objective type {otype}")

    sup = spec.get("support", []) or []
    if not sup:
        support, sa, sb = "R", None, None
    elif len(sup) == 1:
        lhs = _poly(str(sup[0]).split(">=")[0], psyms)
        if sp.degree(lhs, X) != 1:
            raise ValueError("support inequality must be linear (univariate cells)")
        lc = sp.LC(lhs, X)
        root = sp.Rational(sp.solve(lhs, X)[0])
        if lc > 0:
            support, sa, sb = "half_line", root, None       # [a, inf)
        else:
            raise ValueError("upper half-line support not used in these cells")
    elif len(sup) == 2:
        roots = sorted(sp.Rational(sp.solve(_poly(str(s).split(">=")[0], psyms), X)[0]) for s in sup)
        support, sa, sb = "interval", roots[0], roots[1]
    else:
        raise ValueError("at most two support inequalities (univariate)")

    degree = int(spec.get("degree_poly", 0)) or max(
        [int(sp.degree(c.poly, X)) for c in cons] + [2])

    truth = None
    if spec.get("truth"):
        truth = sp.sympify(spec["truth"], locals=psyms, rational=True)

    return LemmaInstance(
        lemma_id=spec["id"], mode=spec["mode"],
        support=support, support_a=sa, support_b=sb,
        constraints=cons, sense=sense, obj_type=otype,
        event_t=event_t, obj_poly=obj_poly, degree=degree,
        params=params, truth=truth, tags=spec.get("tags", []) or [],
    )


def grid_points(spec: dict) -> list[dict[str, Any]]:
    """Cartesian product of parameter grids."""
    import itertools
    pdefs = spec.get("params", {}) or {}
    names = sorted(pdefs)
    grids = [pdefs[n].get("grid", [pdefs[n].get("default")]) for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*grids)]
