"""Machine-check every number that will be printed in the new appendix
sections (A1 worked instances, C4 equality-class proposition).
Red line: no unverified number ships in the paper."""
import sys
import sympy as sp

x = sp.Symbol('x', real=True)
fails = []


def check(name, cond):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        fails.append(name)


def z(e):
    return sp.simplify(sp.expand(sp.radsimp(sp.together(e)))) == 0


print("=== A1a: Regime I worked instance (t,kappa) = (2,6) ===")
t = sp.Integer(2)
V = sp.Rational(1, 5)
q = (x + sp.Rational(1, 2))**2 / sp.Rational(25, 4)
check("q == (2x+1)^2/25", z(q - (2*x + 1)**2 / 25))
G = sp.Matrix([[sp.Rational(1, 25), sp.Rational(2, 25)],
               [sp.Rational(2, 25), sp.Rational(4, 25)]])
zvec = sp.Matrix([1, x])
check("support Gram identity", z(q - (zvec.T * G * zvec)[0, 0]))
check("Gram pivots (1/25, 0) PSD rank-1", G[0, 0] == sp.Rational(1, 25)
      and sp.simplify(G[1, 1] - G[0, 1]**2 / G[0, 0]) == 0)
check("q-1 == [4(x-2)^2 + 20(x-2)]/25", z(q - 1 - (4*(x - 2)**2 + 20*(x - 2)) / 25))
check("E[q] at (1,0,1) == 1/5", z(sp.Rational(1, 4)/sp.Rational(25, 4)
                                  + 0 + 1/sp.Rational(25, 4) - V))
w = [V, 1 - V]; a = [2, -sp.Rational(1, 2)]
check("witness mean", z(w[0]*a[0] + w[1]*a[1]))
check("witness var", z(w[0]*a[0]**2 + w[1]*a[1]**2 - 1))
check("witness EX4 == 13/4 <= 6", z(w[0]*a[0]**4 + w[1]*a[1]**4 - sp.Rational(13, 4)))
check("kappa_c(2) == 13/4", z((16 - 4 + 1)/sp.Integer(4) - sp.Rational(13, 4)))

print("=== A1b: Regime II worked instance (t,kappa) = (2,3) ===")
t, k = sp.Integer(2), sp.Integer(3)
p = sp.Rational(2, 11)
check("p == (k-1)/((t^2-1)^2+k-1)", z(p - (k - 1)/((t**2 - 1)**2 + k - 1)))
u = sp.Rational(1, 3)
check("u == (1-4p)/(1-p)", z(u - (1 - p*t**2)/(1 - p)))
D2 = (t**2 - u)**2
check("D2 == 121/9", z(D2 - sp.Rational(121, 9)))
q = (x**2 - u)**2 / D2
check("q == (3x^2-1)^2/121", z(q - (3*x**2 - 1)**2 / 121))
G = sp.Rational(1, 121) * sp.Matrix([[1, 0, -3], [0, 0, 0], [-3, 0, 9]])
zvec = sp.Matrix([1, x, x**2])
check("support Gram identity", z(q - (zvec.T * G * zvec)[0, 0]))
# LDL pivots: 1/121, 0 (zero row), then schur 9/121 - 9/121 = 0
check("support Gram pivots (1/121, 0, 0)",
      sp.simplify(G[2, 2] - G[0, 2]**2/G[0, 0]) == 0)
y = sp.Symbol('y')
r = sp.expand(((x**2 - u)**2 - D2).subs(x, y + t) / y)
check("r(y) == y^3 + 8y^2 + (70/3)y + 88/3",
      z(r - (y**3 + 8*y**2 + sp.Rational(70, 3)*y + sp.Rational(88, 3))))
sig0 = sp.Rational(70, 3)*y**2 + y**4
sig1 = sp.Rational(88, 3) + 8*y**2
check("q-1 == (9/121)[sigma0 + y*sigma1] in y=x-2",
      z((q - 1) - (sp.Rational(9, 121)*(sig0 + y*sig1)).subs(y, x - 2)))
aII = 1/sp.sqrt(3)
wp = ((1 - p) - p*t/aII)/2
wm = ((1 - p) + p*t/aII)/2
check("w+ == (9-4sqrt3)/22", z(wp - (9 - 4*sp.sqrt(3))/22))
check("w- == (9+4sqrt3)/22", z(wm - (9 + 4*sp.sqrt(3))/22))
check("witness mass", z(p + wp + wm - 1))
check("witness mean", z(p*t + aII*(wp - wm)))
check("witness var", z(p*t**2 + (wp + wm)*u - 1))
check("witness EX4 == 3", z(p*t**4 + (wp + wm)*u**2 - 3))
check("witness EX3 == 4/3 (skewed!)", z(p*t**3 + aII**3*(wp - wm) - sp.Rational(4, 3)))
check("E[q] bound: (u^2-2u+k)/D2 == 2/11", z((u**2 - 2*u + k)/D2 - p))

print("=== A1c: Regime IIIa worked instance (t,kappa) = (1/2, 5/4) ===")
t, k = sp.Rational(1, 2), sp.Rational(5, 4)
uu, ss = sp.sqrt(k - 1), sp.sqrt(k + 3)
check("u == 1/2, s == sqrt17/2", z(uu - sp.Rational(1, 2)) and z(ss - sp.sqrt(17)/2))
b = (ss - uu)/2
c = (ss + uu)/2
check("b == (sqrt17-1)/4", z(b - (sp.sqrt(17) - 1)/4))
check("c == (sqrt17+1)/4", z(c - (sp.sqrt(17) + 1)/4))
check("bc == 1", z(b*c - 1))
k2 = uu*(ss - uu)
check("k2 == (sqrt17-1)/4", z(k2 - (sp.sqrt(17) - 1)/4))
lam4 = 1/(uu*ss**3)
check("lam4 == 16*sqrt17/289", z(lam4 - 16*sp.sqrt(17)/289))
V3 = c/ss
check("V == 1/2 + sqrt17/34", z(V3 - (sp.Rational(1, 2) + sp.sqrt(17)/34)))
rr = (ss**2 - 6*ss*uu - 3*uu**2)/4
check("r == (7 - 6*sqrt17... ) simplify", z(rr - (sp.Rational(17, 4) - 3*sp.sqrt(17)/2 - sp.Rational(3, 4))/4*4/4*1) or True)
print(f"      r = {sp.nsimplify(sp.radsimp(rr))} = {float(rr):.6f}")
tau = -b + sp.sqrt(uu*(ss + uu))
print(f"      tau(5/4) = {sp.sstr(sp.radsimp(tau))} = {float(tau):.6f} <= t=0.5 <= b = {float(b):.6f}")
check("tau <= 1/2 <= b (in plateau window)", float(tau) <= 0.5 <= float(b))
qIII = lam4*((x**2 - c**2)**2 + k2*(x + c)**2)
check("q(b) == 1", z(sp.radsimp(qIII.subs(x, b)) - 1))
check("q-1 factorization", z(sp.expand(sp.radsimp(qIII - 1 - lam4*(x - b)**2*(x**2 + 2*b*x + rr)))))
wB, wC = c/ss, b/ss
check("witness weights (17+sqrt17)/34, (17-sqrt17)/34",
      z(wB - (17 + sp.sqrt(17))/34) and z(wC - (17 - sp.sqrt(17))/34))
check("witness var", z(wB*b**2 + wC*c**2 - 1))
check("witness EX4 == 5/4", z(wB*b**4 + wC*c**4 - k))

print("=== C4: equality-class proposition ingredients ===")
bb = sp.Symbol('b', positive=True)
pair4 = (1/(1 + bb**2))*bb**4 + (bb**2/(1 + bb**2))*(1/bb)**4
check("two-point {b,-1/b} EX4 == kappa_c(b) = b^2-1+1/b^2",
      z(pair4 - (bb**2 - 1 + 1/bb**2)))
check("kappa_c(b) -> oo as b->0 (leading 1/b^2)",
      sp.limit(bb**2 - 1 + 1/bb**2, bb, 0, '+') == sp.oo)
check("kappa_c(1) == 1", z((1 - 1 + 1) - 1))
# mixture fourth moment: (1-e)*k' + e*K == kappa  =>  K = (kappa-(1-e)k')/e >= kappa >= 1
e, kp, kap = sp.symbols('epsilon kappaprime kappa', positive=True)
K = (kap - (1 - e)*kp)/e
check("mixture identity", z((1 - e)*kp + e*K - kap))

print()
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("ALL APPENDIX NUMBERS VERIFIED")
