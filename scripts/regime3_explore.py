"""Regime III (t<1, kappa<kappa_c): solve the extremal system at high precision,
cross-check against SDP, and attempt recognition of the value.

Structure hypothesis (from dual analysis):
  extremal atoms {-c, t, b} with c > b > t, weights w0, w1, w2;
  dual q(x) = lam4 * ((x^2-c^2)^2 + k2*(x+c)^2), k2 = 2 b (c - b),
  with q(t) = q(b) = 1, q'(b) = 0.
System: moment equations (mass, mean, var, kurt binding) + E1==E2:
  (t^2-c^2)^2 + 2 b (c-b) (t+c)^2 == (c-b)(b+c)^3.
Value V = w1 + w2; dual value must match: lam4*(c^4+k2*c^2 + k2 - 2c^2 + kappa).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpmath import mp, mpf, findroot, pslq

mp.dps = 50


def solve_regime3(t, kappa, guess=None):
    t = mpf(t); kappa = mpf(kappa)

    def eqs(w0, w1, w2, c, b):
        return [
            w0 + w1 + w2 - 1,
            -c*w0 + t*w1 + b*w2,
            c*c*w0 + t*t*w1 + b*b*w2 - 1,
            c**4*w0 + t**4*w1 + b**4*w2 - kappa,
            (t*t - c*c)**2 + 2*b*(c - b)*(t + c)**2 - (c - b)*(b + c)**3,
        ]

    if guess is None:
        guess = [mpf('0.3'), mpf('0.2'), mpf('0.5'), mpf('1.6'), mpf('0.7')]
    sol = findroot(eqs, guess)
    w0, w1, w2, c, b = [sol[i] for i in range(5)]
    V = w1 + w2
    k2 = 2*b*(c - b)
    lam4 = 1/((c - b)*(b + c)**3)
    Vdual = lam4*(c**4 + k2*c*c + k2 - 2*c*c + kappa)
    return dict(w0=w0, w1=w1, w2=w2, c=c, b=b, V=V, Vdual=Vdual, k2=k2, lam4=lam4)


if __name__ == "__main__":
    for (t, kappa, sdp) in [("0.5", "1.25", 0.621267813),
                            ("0.5", "1.5", 0.667242152),
                            ("0.5", "2", 0.726255879),
                            ("0.5", "3", 0.789626515)]:
        try:
            r = solve_regime3(t, kappa)
            print(f"t={t} kappa={kappa}:")
            print(f"  V      = {mp.nstr(r['V'], 30)}   (SDP {sdp})")
            print(f"  Vdual  = {mp.nstr(r['Vdual'], 30)}")
            print(f"  atoms: -c={mp.nstr(-r['c'],20)}, t={t}, b={mp.nstr(r['b'],20)}")
            print(f"  wts  : {mp.nstr(r['w0'],15)}, {mp.nstr(r['w1'],15)}, {mp.nstr(r['w2'],15)}")
            # recognition attempts on V
            v = r['V']
            rel = pslq([mpf(1), v], maxcoeff=10**10)
            print(f"  pslq[1,v] (rational?): {rel}")
            rel2 = pslq([mpf(1), v, v*v], maxcoeff=10**8)
            print(f"  pslq[1,v,v^2] (quadratic?): {rel2}")
            rel3 = pslq([mpf(1), v, v*v, v**3], maxcoeff=10**6)
            print(f"  pslq[1,v,v^2,v^3] (cubic?): {rel3}")
        except Exception as e:
            print(f"t={t} kappa={kappa}: FAILED {type(e).__name__}: {e}")
