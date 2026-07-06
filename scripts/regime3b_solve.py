"""Regime IIIb: 3 atoms {-c, t, b}; dual q = lam4*((x^2-c^2)^2 + k2*(x+c)^2).
Full system (7 unknowns): w0,w1,w2 (weights), c, b, k2, lam4.
Equations: 4 moment equations (mass, mean, var, kurt=kappa binding),
q(t)=1, q(b)=1, q'(b)=0. [q(-c)=q'(-c)=0 hold by construction.]
Cross-checks: V=w1+w2 equals dual value lam0+lam2+lam4*kappa; weights >= 0;
q-1 >= 0 on [t,inf); q >= 0 on R.

Then: recognize V and the IIIa/IIIb boundary tau(kappa).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpmath import mp, mpf, findroot, pslq, sqrt

mp.dps = 50


def q_poly(x, c, k2, lam4):
    return lam4 * ((x**2 - c**2)**2 + k2 * (x + c)**2)


def dq_poly(x, c, k2, lam4):
    return lam4 * (4 * x * (x**2 - c**2) + 2 * k2 * (x + c))


def solve_IIIb(t, kappa, guess):
    t = mpf(t); kappa = mpf(kappa)

    def eqs(w0, w1, w2, c, b, k2, lam4):
        return [
            w0 + w1 + w2 - 1,
            -c*w0 + t*w1 + b*w2,
            c*c*w0 + t*t*w1 + b*b*w2 - 1,
            c**4*w0 + t**4*w1 + b**4*w2 - kappa,
            q_poly(t, c, k2, lam4) - 1,
            q_poly(b, c, k2, lam4) - 1,
            dq_poly(b, c, k2, lam4),
        ]

    sol = findroot(eqs, guess)
    w0, w1, w2, c, b, k2, lam4 = [sol[i] for i in range(7)]
    V = w1 + w2
    # dual value: q = lam4*x^4 + 0*x^3 + lam4*(k2-2c^2)*x^2 + 2*lam4*k2*c*x + lam4*(c^4+k2*c^2)
    lam0 = lam4 * (c**4 + k2 * c**2)
    lam2 = lam4 * (k2 - 2 * c**2)
    Vdual = lam0 + lam2 + lam4 * kappa
    return dict(w0=w0, w1=w1, w2=w2, c=c, b=b, k2=k2, lam4=lam4, V=V, Vdual=Vdual)


def try_recognize(name, v):
    for label, vec, mc in [("rational", [mpf(1), v], 10**9),
                           ("quad", [mpf(1), v, v*v], 10**7),
                           ("cubic", [mpf(1), v, v*v, v**3], 10**5)]:
        rel = pslq(vec, maxcoeff=mc, maxsteps=10**4)
        if rel:
            print(f"    {name} {label}: {rel}")
            return rel
    print(f"    {name}: no low-degree relation found")
    return None


if __name__ == "__main__":
    cases = [
        ("1/2", "2", [mpf('0.2737'), mpf('0.25'), mpf('0.4763'), mpf('1.607'), mpf('0.9515'), mpf('1.0'), mpf('0.1')]),
        ("1/2", "3", [mpf('0.2104'), mpf('0.3'), mpf('0.4896'), mpf('1.930'), mpf('1.190'), mpf('1.5'), mpf('0.05')]),
    ]
    for tt, kk, guess in cases:
        r = solve_IIIb(tt, kk, guess)
        print(f"t={tt} kappa={kk}:")
        print(f"  V     = {mp.nstr(r['V'], 30)}")
        print(f"  Vdual = {mp.nstr(r['Vdual'], 30)}")
        print(f"  atoms: -c={mp.nstr(-r['c'],20)}  t  b={mp.nstr(r['b'],20)}")
        print(f"  wts:   {mp.nstr(r['w0'],12)} {mp.nstr(r['w1'],12)} {mp.nstr(r['w2'],12)}")
        print(f"  k2={mp.nstr(r['k2'],20)} lam4={mp.nstr(r['lam4'],20)}")
        try_recognize("V", r['V'])
        try_recognize("c", r['c'])
        try_recognize("b", r['b'])

    # IIIa boundary tau(kappa): with bc=1, b+c=sqrt(kappa+3), k2=2b(c-b),
    # lam4 = 1/((c-b)(b+c)^3); other two roots of q-1: r1+r2=-2b, r1*r2 from const term.
    print("\nIIIa boundary tau(kappa):")
    for kk in ["5/4", "3/2", "2", "3"]:
        kap = mpf(kk)
        S = sqrt(kap + 3)
        d = sqrt(kap - 1)
        b = (S - d) / 2   # positive atom (small)
        c = (S + d) / 2   # negative atom magnitude
        k2 = 2 * b * (c - b)
        lam4 = 1 / ((c - b) * (b + c)**3)
        # q(0) - 1 = lam4*c^2*(c^2+k2) - 1 = lam4 * b^2 * r1 * r2
        r1r2 = (c**2 * (c**2 + k2) - 1/lam4) / b**2
        # r1 + r2 = -2b
        disc = b*b - r1r2
        tau = -b + sqrt(disc)
        print(f"  kappa={kk}: b={mp.nstr(b,12)} c={mp.nstr(c,12)} tau={mp.nstr(tau, 25)}")
        try_recognize("tau", tau)
        try_recognize("tau^2", tau*tau)
