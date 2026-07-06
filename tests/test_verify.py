"""Known-answer + mutation tests for the independent verifier.

The verifier must (a) PASS the hand-built Appendix-A Cantelli certificate,
and (b) FAIL every mutated version. A verifier that cannot reject bad
certificates is a rubber stamp (red line 3).
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.verify import verify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    with open(os.path.join(HERE, "cantelli_appendixA.cert.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_appendix_a_passes():
    rep = verify(load())
    assert rep["pass"], rep
    assert rep["verdict"] == "VERIFIED-TIGHT"


def test_wrong_bound_fails():
    c = load()
    c["bound_value"] = "1/3"
    assert not verify(c)["pass"]


def test_non_psd_gram_fails():
    c = load()
    # flip a sign so the support Gram becomes indefinite
    c["pieces"][1]["blocks"][0]["gram"] = [["1/4", "1/2"], ["1/2", "1/4"]]
    assert not verify(c)["pass"]


def test_broken_identity_fails():
    c = load()
    c["pieces"][0]["blocks"][1]["gram"] = [["2"]]  # sigma_1 = 2 instead of 1
    assert not verify(c)["pass"]


def test_disallowed_multiplier_fails():
    c = load()
    c["pieces"][1]["blocks"][0]["multiplier"] = "x - 1"  # not a generator of R
    assert not verify(c)["pass"]


def test_infeasible_witness_fails():
    c = load()
    c["extremal"]["atoms"] = ["1", "-2"]  # breaks E[x]=0 and E[x^2]=1
    assert not verify(c)["pass"]


def test_float_smuggling_fails():
    c = load()
    c["dual_multipliers"] = [0.25, "1/2", "1/4"]  # float not allowed
    assert not verify(c)["pass"]


def test_wrong_multiplier_sign_fails():
    # turn variance constraint into an inequality with a wrong-signed multiplier
    c = load()
    c["constraints"][2]["op"] = ">="          # E[x^2] >= 1 in a sup problem
    assert not verify(c)["pass"]              # needs lambda <= 0, but it is 1/4 > 0


def test_float_in_gram_fails():
    c = load()
    c["pieces"][1]["blocks"][0]["gram"] = [[0.25, "1/4"], ["1/4", "1/4"]]
    assert not verify(c)["pass"]
