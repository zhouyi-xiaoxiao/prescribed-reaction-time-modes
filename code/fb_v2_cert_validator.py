#!/usr/bin/env python3
"""Theorem-level validator (acceptance test) for the item-3 interval certificates.

The per-box flags of fb_v2_fold_curve_certificate.py (CV-1) and fb_v2_codim2_switch.py (CV-2) are
narrower than the propositions that cite them (theory/V2_fold_curve_m2.tex Prop. v2:prop-foldcurve,
theory/V2_codim2_switch.tex Prop. v2:prop-alloc).  This script is the ACCEPTANCE TEST of those
propositions: it exits with status 0 only if every claim passes.  The exit status of the generators is
not theorem acceptance.  Every decimal string of the JSONs is a certified one-sided bound; Fraction(s) is
its exact value, and all pass/fail decisions are exact rational comparisons (or mpmath.iv, 100 bits,
where a transcendental function of stored bounds is needed).

CV-1 (``--only cv1`` needs only the CV-1 JSONs and the TH8 anchor certificate):
  C0 provenance: the JSONs carry the sha256 of the generator sources that produced them, and these equal
     the hashes of the files on disk (the run is tied to the audited source); mpmath version recorded;
  C1 the certified set: exact partition invariants, certified unions [0.15, 0.40] and [0.40, 0.5715],
     no excluded interval, and the subdivision facts quoted in the proof (250 base boxes of width 1e-3 at
     depth 0; 343 base boxes of width 5e-4, of which exactly the last two are bisected, to depth <= 3,
     347 leaves), and the stored summaries agree with the box records;
  C2 every per-box condition ((D1)-(D4), Krawczyk zero in the (D3) region, 1-D Newton, a_f > 0, strict
     monotonicity d log b_2/d a > 0 and dB/drho < 0);
  C3 (D4) margin log r0(tau) - log b_2 >= 5.66;  C4 B_top^det in [1.64115, 5581.84] on [0.15, 0.40];
  C5 tube accuracy 3.9e-4, with the endpoint tubes REBUILT from the stored coefficients (K_m, S, exact
     dyadic a_m, Delta enclosure) in interval arithmetic (convexity of the width in a - a_m);
  C6 at exactly the six widths 0.15, ..., 0.40: certified thin boxes whose B_top^det, t_fold and a_f
     enclosures are string-identical to TH8_certificate/certificate.json (read directly); a missing
     width or quantity fails.
CV-2 (``--only cv2``): (i) F2W, (ii) max-min path, (iii) F3S switch -- including the end-sign evidence
  (rational comparisons of the stored b_2, b_3 enclosures at exact dyadic end points inside the exact
  q-box of the thick certificate), the claimed derivative domain [1.52, 1.57], absence of a failure
  record, a_f2, a_f3 > 0, and the 4-D Krawczyk cross-check with its identification replayed on exact
  dyadic domains --, (iv) F3A, and the equal-weight anchor.
Reported witnesses are exact decimal strings or DIRECTED decimals (``*_lower`` rounded down,
``*_upper`` rounded up); ``approx_*`` fields are diagnostics only.
Also a float (mpmath, 30 digits, NOT a certificate) spot check of the analytic lemma behind the
equal-weight no-cusp statement: every zero of Phi on (-u*, 0) has Phi_u > 0.

Run (from code/):  python fb_v2_cert_validator.py [--only cv1|cv2]
Output: ../artifacts/data/exact_m_fixed_budget/V2_bifurcation/theorem_validation.json
        (theorem_validation_cv1.json / theorem_validation_cv2.json with --only)
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget"
BIF = DATA / "V2_bifurcation"
TH8 = DATA / "TH8_certificate" / "certificate.json"
OUT = BIF / "theorem_validation.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def contains(outer, inner) -> bool:
    """claimed interval outer (decimal strings) contains the certified enclosure inner."""
    return F(outer[0]) <= F(inner[0]) and F(inner[1]) <= F(outer[1])


# ---------------------------------------------------------------------------- exact / directed decimals
def _shift_str(n: int, k: int) -> str:
    """exact decimal string of n * 10^(-k)."""
    neg, n = n < 0, abs(n)
    if k <= 0:
        s = str(n * 10 ** (-k))
    else:
        d = str(n).rjust(k + 1, "0")
        s = (d[:-k] + "." + d[-k:]).rstrip("0").rstrip(".")
    return ("-" if neg else "") + s


def exact_dec(q) -> str:
    """exact decimal string of a rational with terminating expansion (else 'p/q')."""
    q = F(q)
    for k in range(0, 400):
        if (q.denominator and (10 ** k) % q.denominator == 0):
            return _shift_str(q.numerator * (10 ** k // q.denominator), k)
    return f"{q.numerator}/{q.denominator}"


def directed(q, upper: bool, sig: int = 12) -> str:
    """decimal string with `sig` significant digits, rounded DOWN (upper=False) or UP (upper=True)."""
    q = F(q)
    if q == 0:
        return "0"
    a = abs(q)
    e = len(str(a.numerator)) - len(str(a.denominator))
    while F(10) ** e > a:
        e -= 1
    while F(10) ** (e + 1) <= a:
        e += 1
    k = sig - 1 - e
    x = q * F(10) ** k
    n = -((-x.numerator) // x.denominator) if upper else x.numerator // x.denominator
    return _shift_str(n, k)


def lower(q, sig=12) -> str:
    return directed(q, False, sig)


def upper(q, sig=12) -> str:
    return directed(q, True, sig)


def mpf_to_frac(x) -> F:
    """exact rational value of a binary mpf."""
    sign, man, ex, _ = x._mpf_
    v = F(int(man)) * (F(2) ** ex if ex >= 0 else F(1, 2 ** (-ex)))
    return -v if sign else v


def partition(boxes, a: str, b: str, key) -> dict:
    """exact partition invariants of a list of box records over [a, b]."""
    ivs = [key(r) for r in boxes]
    ok_sorted = all(F(ivs[i][0]) < F(ivs[i][1]) for i in range(len(ivs)))
    ok_shared = all(ivs[i][1] == ivs[i + 1][0] for i in range(len(ivs) - 1))
    ok_ends = bool(ivs) and F(ivs[0][0]) == F(a) and F(ivs[-1][1]) == F(b)
    cert, excl = [], []
    for r, (x, y) in zip(boxes, ivs):
        tgt = cert if r["certified"] else excl
        if tgt and tgt[-1][1] == x:
            tgt[-1][1] = y
        else:
            tgt.append([x, y])
    return {"n_boxes": len(ivs), "boxes_nondegenerate": ok_sorted, "consecutive_ends_shared_exactly": ok_shared,
            "ends_equal_claim": ok_ends, "certified_unions": cert, "excluded_unions": excl,
            "all_ok": ok_sorted and ok_shared and ok_ends}


def check(name, claim, passed, **values):
    return {"claim": claim, "pass": bool(passed), **values}


def provenance_check(J, files) -> dict:
    """the JSON's recorded source hashes equal the hashes of the generator files on disk."""
    prov = J.get("provenance") or {}
    rec = prov.get("source_sha256") or {}
    now = {f: sha256(HERE / f) for f in files}
    match = {f: (rec.get(f) == now[f]) for f in files}
    return {"recorded_mpmath_version": prov.get("mpmath_version"), "recorded_python_version": prov.get("python_version"),
            "source_sha256_on_disk": now, "source_sha256_matches_json": match,
            "ok": bool(prov) and all(match.values())}


# ============================================================================
# CV-1
# ============================================================================
SIX = ("0.15", "0.2", "0.25", "0.3", "0.35", "0.4")


def subdivision(leaves, a: str, w: F, n_base: int) -> dict:
    """reconstruct the base boxes [a + i w, a + (i+1) w] from the leaves (exact)."""
    A = F(a)
    per, ok = {}, True
    for r in leaves:
        x, y, dpt = F(r["rho_box"][0]), F(r["rho_box"][1]), r["depth"]
        i = (x - A) // w
        ok &= (y - x) == w / 2 ** dpt and A + i * w <= x and y <= A + (i + 1) * w
        per.setdefault(int(i), []).append(dpt)
    return {"base_indices_complete": sorted(per) == list(range(n_base)), "leaf_widths_consistent": bool(ok),
            "bisected_base_boxes": sorted(i for i, v in per.items() if len(v) > 1),
            "max_depth": max(r["depth"] for r in leaves), "n_leaves": len(leaves)}


def validate_cv1() -> dict:
    J = json.loads((BIF / "fold_curve_m2.json").read_text())
    BX = json.loads((BIF / "fold_curve_m2_boxes.json").read_text())
    th8 = json.loads(TH8.read_text())["cases"]
    req, ext = BX["required_range"], BX["extension_towards_rho_c"]
    out = {"source": "V2_bifurcation/fold_curve_m2.json (+ _boxes.json), TH8_certificate/certificate.json",
           "theory": "theory/V2_fold_curve_m2.tex Prop. v2:prop-foldcurve",
           "input_sha256": {p: sha256(BIF / p) for p in ("fold_curve_m2.json", "fold_curve_m2_boxes.json")}
           | {"TH8_certificate/certificate.json": sha256(TH8)}}
    pc = provenance_check(J, ("fb_v2_fold_curve_certificate.py", "fb_th8_interval_certificate.py"))
    out["C0_provenance"] = check("C0", "the certificate JSON records the sha256 of the generator sources, equal to "
                                       "the files on disk", pc["ok"], **pc)
    key = lambda r: r["rho_box"]                                     # noqa: E731
    p_req, p_ext = partition(req, "0.15", "0.4", key), partition(ext, "0.4", "0.5715", key)
    sub_req = subdivision(req, "0.15", F(1, 1000), 250)
    sub_ext = subdivision(ext, "0.4", F(1, 2000), 343)
    sub_ok = (sub_req["base_indices_complete"] and sub_req["leaf_widths_consistent"] and sub_req["max_depth"] == 0
              and sub_req["n_leaves"] == 250
              and sub_ext["base_indices_complete"] and sub_ext["leaf_widths_consistent"]
              and sub_ext["bisected_base_boxes"] == [341, 342] and sub_ext["max_depth"] <= 3
              and sub_ext["n_leaves"] == 347)
    sr, se = J["required_range"], J["extension_towards_rho_c"]
    summ_ok = (sr["n_leaf_boxes"] == len(req) == sr["n_certified"] and sr["coverage_fraction_exact"] == "1"
               and sr["excluded_intervals"] == [] and se["n_leaf_boxes"] == len(ext) == se["n_certified"]
               and se["coverage_fraction_exact"] == "1" and se["excluded_intervals"] == []
               and J.get("complete_certified_coverage") is True)
    out["C1_set"] = check("C1", "certified set = [0.15, 0.40] U [0.40, 0.5715], every box certified; 250 base boxes "
                                "(width 1e-3, depth 0) and 343 base boxes (width 5e-4), exactly the last two "
                                "bisected, depth <= 3, 347 leaves; stored summaries agree with the box records",
                          p_req["all_ok"] and p_ext["all_ok"] and p_req["certified_unions"] == [["0.15", "0.4"]]
                          and p_ext["certified_unions"] == [["0.4", "0.5715"]]
                          and not p_req["excluded_unions"] and not p_ext["excluded_unions"] and sub_ok and summ_ok,
                          required=p_req, extension=p_ext, subdivision_required=sub_req,
                          subdivision_extension=sub_ext, summaries_consistent=summ_ok)
    per_box, worst = [], {"D4_gap": None, "a_f": None, "slope": None, "dB_drho": None}
    all_ok = True
    for r in req + ext:
        f = r.get("fold", {})
        tb = f.get("tube") or {}
        conds = {"certified_flag": r.get("certified") is True,
                 "D1": r["D1"]["holds"] is True, "D2": r["D2"]["holds"] is True, "D3": r["D3"]["holds"] is True,
                 "D4": r["D4"]["holds"] is True,
                 "krawczyk_in_D3_region": f.get("krawczyk_box_in_D3_region") is True,
                 "newton_1d": f.get("newton_1d_ok") is True,
                 "a_f_positive": F(f["a_f=-r0pp(t_f)"][0]) > 0,
                 "tube": bool(tb),
                 "dlogB_da_positive": bool(tb) and F(tb["dlogB_da_on_box"][0]) > 0,
                 "dB_drho_negative": bool(tb) and F(tb["dB_drho_on_box"][1]) < 0,
                 "D4_margin_positive": F(r["D4"]["log_r0_tau_lower_bound"]) > F(r["D4"]["log_b2_upper_bound"])}
        vals = {"D4_gap": F(r["D4"]["log_r0_tau_lower_bound"]) - F(r["D4"]["log_b2_upper_bound"]),
                "a_f": F(f["a_f=-r0pp(t_f)"][0]), "slope": F(tb["dlogB_da_on_box"][0]),
                "dB_drho": F(tb["dB_drho_on_box"][1])}
        for k, v in vals.items():
            if worst[k] is None or (v < worst[k] if k != "dB_drho" else v > worst[k]):
                worst[k] = v
        ok = all(conds.values())
        all_ok &= ok
        if not ok:
            per_box.append({"rho_box": r["rho_box"], "failed": [k for k, v in conds.items() if not v]})
    out["C2_per_box_conditions"] = check(
        "C2", "(D1)-(D4), Krawczyk zero in the (D3) region, 1-D Newton, a_f > 0, and strict monotonicity "
              "(d log b_2/d a > 0, d B/d rho < 0) on every box of [0.15, 0.5715]", all_ok,
        failures=per_box, min_D4_log_gap_exact=exact_dec(worst["D4_gap"]), min_a_f_lower=worst_str(worst["a_f"]),
        min_dlogB_da_lower=worst_str(worst["slope"]), max_dB_drho_upper=worst_str(worst["dB_drho"]))
    gap_min = worst["D4_gap"]
    gap_sum = min(F(J["required_range"]["D4_log_gap_lower_bound_min"]),
                  F(J["extension_towards_rho_c"]["D4_log_gap_lower_bound_min"]))
    out["C3_D4_gap"] = check("C3", "log r0(tau) - log b_2 >= 5.66 on every box (proof text)",
                             gap_min >= F("5.66") and gap_sum >= F("5.66"),
                             min_gap_exact=exact_dec(gap_min), min_gap_lower=lower(gap_min),
                             min_interval_gap_lower_bound=exact_dec(gap_sum))
    Blo_s = min((r["fold"]["B_top_det"][0] for r in req), key=F)
    Bhi_s = max((r["fold"]["B_top_det"][1] for r in req), key=F)
    hull_ok = [F(x) for x in J["required_range"]["B_top_det_hull_over_range"]] == [F(Blo_s), F(Bhi_s)]
    out["C4_B_range"] = check("C4", "B_top^det(rho) in [1.64115, 5581.84] for rho in [0.15, 0.40]",
                              F(Blo_s) >= F("1.64115") and F(Bhi_s) <= F("5581.84") and hull_ok,
                              hull_exact=[Blo_s, Bhi_s], summary_hull_equal=hull_ok)
    out["C5_tube"] = tube_check(J, req)
    # C6: anchors against the authoritative TH8 certificate (read directly)
    anc = J.get("anchor_consistency_vs_TH8_certificate") or {}
    ident, ok6 = {}, set(anc) == set(SIX)
    for rho in SIX:
        a = anc.get(rho) or {}
        t = th8.get(f"m2_rho{rho}") or {}
        th_B = t.get("B_top_det")
        th_t = ((t.get("D3") or {}).get("flanks") or [{}])[0].get("t_fold")
        th_a = (t.get("fold") or {}).get("a_f=-r0pp(t_f)")
        row = {"certified_thin": a.get("certified_thin") is True,
               "B_top_det": th_B is not None and a.get("B_top_det_uA") == th_B,
               "t_fold": th_t is not None and a.get("t_fold_uA") == th_t,
               "a_f": th_a is not None and a.get("a_f_uA") == th_a,
               "TH8_certified": t.get("certified") is True}
        ident[rho] = row
        ok6 &= all(row.values())
    out["C6_anchor_digits"] = check("C6", "at exactly the six widths 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 the thin boxes are "
                                          "certified and the B_top^det, t_fold and a_f enclosures are string-identical "
                                          "to TH8_certificate/certificate.json (all printed digits); missing = fail",
                                    ok6, identity=ident)
    return out


def worst_str(q) -> str:
    """the extreme value is itself a stored decimal bound: print it exactly."""
    return exact_dec(q)


def tube_check(J, req) -> dict:
    """C5: relative accuracy of the tube rebuilt from the STORED coefficients, at both ends of every box."""
    from mpmath import iv, mp
    mp.prec = iv.prec = 100
    Dl = J["reduction"]["Delta"]
    DEL = iv.mpf([Dl[0], Dl[1]])
    thr = F("3.9e-4")
    worst_exact_am, worst_iv_am, worst_stored, n_exact = F(0), F(0), F(0), 0
    for r in req:
        tb = r["fold"]["tube"]
        Km = iv.mpf([tb["logB_at_a_mid"][0], tb["logB_at_a_mid"][1]])
        S = iv.mpf([tb["dlogB_da_on_box"][0], tb["dlogB_da_on_box"][1]])
        am_iv = iv.mpf([tb["a_mid"][0], tb["a_mid"][1]])
        am_ex = tb.get("a_mid_exact")
        if am_ex is not None:
            am_x = iv.mpf(am_ex)
            if mpf_to_frac(mp.mpf(am_x.a)) != F(am_ex) or mpf_to_frac(mp.mpf(am_x.b)) != F(am_ex):
                raise ValueError("a_mid_exact not exactly representable at 100 bits")
            n_exact += 1
        for name, rho_end in (("at_rho_lo", r["rho_box"][0]), ("at_rho_hi", r["rho_box"][1])):
            worst_stored = max(worst_stored, F(tb["ends"][name]["rel_halfwidth_B_upper"]))
            a_end = DEL / (2 * iv.mpf([rho_end, rho_end]))
            for am, tag in ((am_x, "exact") if am_ex is not None else (None, None), (am_iv, "iv")):
                if am is None:
                    continue
                be = Km + S * (a_end - am)
                w = iv.mpf(be.b) - iv.mpf(be.a)                  # upper bound of the width, outward
                rel = (iv.exp(iv.mpf(w.b)) - 1) / 2
                ub = mpf_to_frac(mp.mpf(rel.b))
                if tag == "exact":
                    worst_exact_am = max(worst_exact_am, ub)
                else:
                    worst_iv_am = max(worst_iv_am, ub)
    ok = (n_exact == len(req) and worst_exact_am <= thr and worst_stored <= thr)
    return check("C5", "relative accuracy 3.9e-4 at every rho in [0.15, 0.40]: the endpoint tubes K_m + S (a(rho_end) "
                       "- a_m) are REBUILT in interval arithmetic from the stored K_m, S, exact dyadic a_m and the "
                       "Delta enclosure; the pointwise width w(K_m) + w(S)|a - a_m| is convex in a, so the box ends "
                       "bound it", ok,
                 max_rebuilt_exact_a_mid_upper=upper(worst_exact_am, 8),
                 max_stored_upper=upper(worst_stored, 8),
                 approx_max_rebuilt_with_interval_a_mid=upper(worst_iv_am, 8),
                 n_boxes_with_exact_a_mid=n_exact,
                 argument="|B - M| <= B (e^w - 1)/2 for M the midpoint of [e^lo, e^hi], w the width of the "
                          "log-enclosure; stored representation: K_m, S (outward decimal intervals), a_m exact")


# ============================================================================
# CV-2
# ============================================================================


def validate_cv2() -> dict:
    J = json.loads((BIF / "codim2_switch.json").read_text())
    BX = json.loads((BIF / "codim2_switch_boxes.json").read_text())
    out = {"source": "V2_bifurcation/codim2_switch.json (+ _boxes.json)",
           "theory": "theory/V2_codim2_switch.tex Prop. v2:prop-alloc",
           "input_sha256": {p: sha256(BIF / p) for p in ("codim2_switch.json", "codim2_switch_boxes.json")}}
    pc = provenance_check(J, ("fb_v2_codim2_switch.py", "fb_v2_fold_curve_certificate.py",
                              "fb_th8_interval_certificate.py"))
    out["P0_provenance"] = check("P0", "the certificate JSON records the sha256 of the generator sources, equal to "
                                       "the files on disk", pc["ok"], **pc)
    key = lambda r: r["box"]                                          # noqa: E731
    # (i) F2W
    L = BX["F2W"]
    part = partition(L, "0.025", "0.975", key)
    cert = [r for r in L if r["certified"]]
    bind = all(r["binding"] == "flank_2" and r["D4_on_box"] for r in cert)
    ratio = min(F(r["r0_tau"][0]) / F(r["flanks"][0]["b"][1]) for r in cert)
    out["i_F2W"] = check("(i)", "for every w2 in [0.02515625, 0.97484375]: (D1)-(D4), rising flank binds, "
                                "B_top^det = b_2 < r0(tau)/547",
                         part["all_ok"] and part["certified_unions"] == [["0.02515625", "0.97484375"]] and bind
                         and ratio > 547, partition=part, min_r0tau_over_b2_lower=lower(ratio, 10),
                         binding_flank2_and_D4_on_every_certified_box=bind)
    # (ii) max-min path
    P = J["maxmin_path_F2W"]
    rec = {tuple(r["box"]): r for r in L}
    w0 = P["w2_star_B_to_0"]
    rows = P["rows"]
    first = rows[0]["w_box"]
    start_ok = F(first[0]) <= F(w0[0]) and F(w0[1]) < F(first[1])
    contiguous = all(rows[i]["w_box"][1] == rows[i + 1]["w_box"][0] for i in range(len(rows) - 1))
    below = True
    for rw in rows:
        r = rec[tuple(rw["w_box"])]
        top = min(F(r["r0_tau"][0]), F(r["flanks"][0]["b"][0]))
        below &= r["certified"] and F(rw["B_path_at_hi_w_upper"]) < top
    reach = P["certified_reach"]
    reach_ok = rows[-1]["w_box"][1] == reach["w2_upper"] and F(reach["B_path_lower"]) >= F("40.5457")
    eq = P["at_equal_weight_fold_budget"]
    wq = eq["w2_star"]
    cov = [c for c in eq["covering_boxes"]]
    covered = (F(cov[0][0]) <= F(wq[0]) and F(wq[1]) <= F(cov[-1][1])
               and all(cov[i][1] == cov[i + 1][0] for i in range(len(cov) - 1))
               and all(rec[tuple(c)]["certified"] and rec[tuple(c)]["binding"] == "flank_2" for c in cov))
    b2 = [min((rec[tuple(c)]["flanks"][0]["b"][0] for c in cov), key=F),
          max((rec[tuple(c)]["flanks"][0]["b"][1] for c in cov), key=F)]
    out["ii_path"] = check("(ii)", "max-min path below the fold for 0 < B <= 40.5457; at B = 8.2247 w2* = 0.87599 and "
                                   "B_top^det(w*) in [23.45, 23.88]",
                           start_ok and contiguous and below and reach_ok and covered
                           and F(b2[0]) >= F("23.45") and F(b2[1]) <= F("23.88")
                           and F("0.875985") <= F(wq[0]) and F(wq[1]) <= F("0.875995"),
                           start_box_contains_w2star_0=start_ok, rows_contiguous=contiguous,
                           every_row_below_fold_exact=below, n_rows=len(rows), reach=reach,
                           Wq=wq, Wq_covered_by_certified_flank2_boxes=covered, b2_at_w2star_exact=b2)
    out["iii_F3S"] = validate_switch(J, BX)
    # (iv) F3A
    La = BX["F3A"]
    parta = partition(La, "0.3492", "2.864", key)
    ca = [r for r in La if r["certified"]]
    ratio_a = min(F(r["flanks"][0]["b"][0]) / F(r["flanks"][1]["b"][1]) for r in ca)
    binda = all(r["binding"] == "flank_3" and r["D4_on_box"] for r in ca)
    out["iv_F3A"] = check("(iv)", "for every q in [0.3492, 2.864]: (D1)-(D4), last flank binds, b2 > 36.7 b3",
                          parta["all_ok"] and parta["certified_unions"] == [["0.3492", "2.864"]] and binda
                          and ratio_a > F("36.7"), partition=parta, min_b2_over_b3_lower=lower(ratio_a, 10))
    # equal-weight anchor
    th8 = json.loads(TH8.read_text())["cases"]["m2_rho0.3"]["B_top_det"]
    eqb = [r for r in cert if F(r["box"][0]) <= F(1, 2) <= F(r["box"][1])]
    out["anchor_F2W"] = check("anchor", "every certified F2W box containing w2 = 1/2 contains B_top^det(0.3) of TH8",
                              bool(eqb) and all(contains(r["flanks"][0]["b"], th8) for r in eqb),
                              boxes=[r["box"] for r in eqb], th8=th8)
    return out


def validate_switch(J, BX) -> dict:
    """(iii) F3S: existence, uniqueness and enclosure of the switch q*, with every advertised check gated."""
    key = lambda r: r["box"]                                          # noqa: E731
    L3 = BX["F3S"]
    part3 = partition(L3, "0.3366", "2.864", key)
    sw = J["codim2_switch_F3S"]
    qs = sw["q_star"]
    ratio3 = min(F(r["r0_tau"][0]) / max(F(f["b"][1]) for f in r["flanks"]) for r in L3 if r["certified"])
    br = sw["bracket_initial"]
    cons = True
    for r in L3:
        a, b = F(r["box"][0]), F(r["box"][1])
        if r["binding"] == "flank_2":
            cons &= b <= F(qs[1])
        elif r["binding"] == "flank_3":
            cons &= a >= F(qs[0])
        elif r["binding"] is None:
            cons &= F(br[0]) <= a and b <= F(br[1])
        else:
            cons = False
    # --- the claimed derivative domain, no failure record, end signs by rational comparison
    domain_ok = br == ["1.52", "1.57"] and "fail" not in sw and sw.get("bracket_is_primary") is True
    ev = sw.get("sign_evidence_at_ends") or {}
    qbox = sw.get("bracket_q_box_exact")
    sign_ok = False
    if ev.get("left", {}).get("certified") and ev.get("right", {}).get("certified") and qbox:
        lf, rt = ev["left"], ev["right"]
        ql, qr = F(lf["q_exact"]), F(rt["q_exact"])
        iso = sw["internal_isolating_bracket"]
        sign_ok = (F(lf["b2"][1]) < F(lf["b3"][0]) and F(rt["b2"][0]) > F(rt["b3"][1])      # D(ql) < 0 < D(qr)
                   and sw["sign_D_at_ends"] == [-1, 1]
                   and F(qbox[0]) <= F("1.52") and F("1.57") <= F(qbox[1])                   # claimed domain
                   and F(qbox[0]) <= ql < qr <= F(qbox[1])                                   # ends in the box
                   and ql <= F(iso[0]) < F(iso[1]) <= qr
                   and F(qs[0]) <= F(iso[0]) and F(iso[1]) <= F(qs[1]))
    nondeg = F(sw["a_f2_lower"]) > 0 and F(sw["a_f3_lower"]) > 0
    # --- 4-D Krawczyk cross-check and its identification, replayed on exact dyadic domains
    k4 = sw.get("krawczyk_4d") or {}
    idn = k4.get("identification") or {}
    flags = ("X_s_inside_S", "X_t2_inside_D3_search_flank2", "X_t3_inside_D3_search_flank3",
             "identified_with_switch")
    k4_flags = k4.get("ok") is True and all(idn.get(f) is True for f in flags) \
        and k4.get("consistent_with_bisection") is True
    ed = idn.get("exact_domains")
    replay = False
    if ed:
        X = {k: [F(v[0]), F(v[1])] for k, v in ed["X"].items()}
        S = [F(ed["S"][0]), F(ed["S"][1])]
        s2 = [F(ed["D3_search_flank2"][0]), F(ed["D3_search_flank2"][1])]
        s3 = [F(ed["D3_search_flank3"][0]), F(ed["D3_search_flank3"][1])]
        Bst = [F(sw["B_star"][0]), F(sw["B_star"][1])]
        replay = (S[0] <= X["s"][0] and X["s"][1] <= S[1] and s2[0] <= X["t2"][0] and X["t2"][1] <= s2[1]
                  and s3[0] <= X["t3"][0] and X["t3"][1] <= s3[1]
                  and not (X["B"][1] < Bst[0] or X["B"][0] > Bst[1]))
    printed_w = F(qs[1]) - F(qs[0])
    claims3 = {"q_star": (["1.54500627518087411", "1.54500627518087698"], qs),
               "B_star": (["554.7235711396622", "554.7235711396623"], sw["B_star"]),
               "t_f2_star": (["1.5159546434889", "1.5159546434890"], sw["t_f2_star"]),
               "t_f3_star": (["2.3400950924049", "2.3400950924050"], sw["t_f3_star"]),
               "dD_ds_on_bracket": (["2117", "5091"], sw["dD_ds_on_bracket"])}
    cont = {k: contains(c, e) for k, (c, e) in claims3.items()}
    passed = (part3["all_ok"] and part3["certified_unions"] == [["0.3366", "2.864"]] and ratio3 > F("1.058")
              and cons and all(cont.values()) and F(sw["dD_ds_on_bracket"][0]) > 0
              and sw["left_endpoint_certified_on_bracket"] is True and sw["unique_in_bracket"] is True
              and sw.get("undetermined_boxes_inside_bracket") is True
              and sw.get("exactly_one_switch_in_certified_set") is True
              and F(br[0]) <= F(qs[0]) and F(qs[1]) <= F(br[1])
              and F(sw["t_f2_star"][1]) < F(sw["t_f3_star"][0])
              and domain_ok and sign_ok and nondeg and k4_flags and replay)
    return check("(iii)", "for every q in [0.3366, 2.864]: (D1)-(D3), r0(tau) > 1.058 max(b2, b3), exactly one zero q* "
                          "of b2 - b3 (enclosures as printed), d(b2-b3)/dq > 0 on [1.52, 1.57] (the thick box of the "
                          "claimed domain, no fallback bracket, no failure record), certified opposite end signs, "
                          "a_f2, a_f3 > 0, middle flank binds for q < q*, last for q > q*, (D4) for q != q*, and the "
                          "4-D Krawczyk cross-check succeeds and is identified with the switch",
                 passed, partition=part3, min_r0tau_over_max_b_lower=lower(ratio3, 10),
                 binding_consistent_with_q_star=cons, printed_enclosures_inside_claims=cont,
                 claimed_domain_and_no_failure=domain_ok, end_signs_rational=sign_ok,
                 a_f2_a_f3_positive=nondeg, krawczyk_4d_flags=k4_flags, krawczyk_4d_exact_replay=replay,
                 printed_q_star_width_exact=exact_dec(printed_w),
                 internal_bracket_width_upper=sw.get("internal_isolating_bracket_width_upper"),
                 note="(D4) for q != q* in the undetermined boxes follows from r0(tau) > b2, b3 on the bracket "
                      "(left_endpoint_certified_on_bracket) and strict monotonicity of b2 - b3")


# ============================================================================
# analytic no-cusp lemma: float spot check (NOT a certificate)
# ============================================================================


def nocusp_spot_check() -> dict:
    from mpmath import mp, mpf, tanh, sech, findroot, e
    mp.dps = 30
    c1, c2 = 4 * e ** -1, 4 * e ** mpf("-2.5")
    kap = (c1 + c2) / (c1 - c2)
    worst, n, bad = None, 0, []
    for i in range(1, 60):
        a = 1 + mpf(i) / 20                                 # a in (1, 3.95]
        k = kap * a
        s = lambda u: -u + a * tanh(a * u)                   # noqa: E731
        su = lambda u: -1 + a * a * sech(a * u) ** 2         # noqa: E731
        suu = lambda u: -2 * a ** 3 * sech(a * u) ** 2 * tanh(a * u)   # noqa: E731
        Phi = lambda u: (k + u) * (su(u) - s(u) ** 2) + s(u)          # noqa: E731
        Phiu = lambda u: 2 * su(u) - s(u) ** 2 + (k + u) * (suu(u) - 2 * s(u) * su(u))  # noqa: E731
        us = findroot(s, (mpf("1e-8"), a), solver="bisect")  # u* in (0, a)
        grid = [-us + (us) * mpf(j) / 400 for j in range(1, 400)]
        vals = [Phi(u) for u in grid]
        for j in range(len(grid) - 1):
            if vals[j] == 0 or vals[j] * vals[j + 1] < 0:
                uf = findroot(Phi, (grid[j], grid[j + 1]), solver="bisect")
                d = Phiu(uf)
                n += 1
                worst = d if worst is None or d < worst else worst
                if d <= 0:
                    bad.append(float(a))
    return {"kind": "float spot check (mpmath 30 digits), NOT a certificate", "a_grid": "1.05..3.95 step 0.05",
            "n_zeros_found": n, "approx_min_Phi_u_at_zeros": float(worst), "violations": bad,
            "lemma": "on (-u*, 0): s < 0; a zero of Phi has s_u - s^2 = -s/(k+u) > 0 and s_uu > 0, so "
                     "Phi_u = 2 s_u - s^2 + (k+u)(s_uu - 2 s s_u) > 0"}


def main(argv=None) -> int:
    import argparse
    import mpmath
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("cv1", "cv2"), default=None,
                    help="validate one certificate (cv1 needs only the CV-1 JSONs and the TH8 certificate)")
    args = ap.parse_args(argv)
    parts = (args.only,) if args.only else ("cv1", "cv2")
    res = {"script": "code/fb_v2_cert_validator.py",
           "item": "V2 item 3: theorem-level acceptance test of Props. v2:prop-foldcurve and v2:prop-alloc",
           "arithmetic": "exact rational comparisons of the stored certified decimal bounds (fractions.Fraction); "
                         "mpmath.iv (100 bits) for the tube exponential; witnesses printed exactly or directed",
           "validator_sha256": sha256(Path(__file__).resolve()), "mpmath_version": mpmath.__version__,
           "python_version": platform.python_version()}
    if "cv1" in parts:
        res["cv1"] = validate_cv1()
    if "cv2" in parts:
        res["cv2"] = validate_cv2()
    if args.only is None:
        res["equal_weight_nocusp_lemma_spot_check"] = nocusp_spot_check()
    flat = [v["pass"] for part in parts for k, v in res[part].items() if isinstance(v, dict) and "pass" in v]
    res["all_claims_pass"] = bool(flat) and bool(all(flat))
    res["n_claims"] = len(flat)
    out = OUT if args.only is None else BIF / f"theorem_validation_{args.only}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({part: {k: v["pass"] for k, v in res[part].items() if isinstance(v, dict) and "pass" in v}
                      for part in parts}, indent=1))
    print("all_claims_pass", res["all_claims_pass"], "->", out.name)
    return 0 if res["all_claims_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
