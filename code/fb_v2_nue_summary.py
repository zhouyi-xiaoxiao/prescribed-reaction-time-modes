#!/usr/bin/env python3
"""NU-E summary (uplift2 item 1, acceptance (c)): collect the Total-Error-Protocol results of every
demonstrated V2 design and write V2_TEP/nue_summary.json (+ figure fb_v2_tep_validation.pdf).

Samples per design (all direct kill, seeds disjoint from the design ensembles):
  primary    V2_TEP/nue_<label>_ladder/tep_ladder_nue_<label>.json  coupled ladder, tag 215, 1e6 walkers,
             paired group bootstrap (fb_v2_tep_ladder.py analyze)
  replicate  V2_TEP/nue_<label>/tep_boot_nue_<label>.json           independent levels, tag 212, 1e6 walkers
             per level, unpaired group bootstrap (fb_v2_tep_ladder.py analyze-indep)
  harness    V2_TEP/nue_<label>/tep_nue_<label>.json                same runs, fb_v2_tep.py jackknife (kept for
             reference; its followed-stationary-point jackknife is unreliable on flat late peaks)
  isambard   V2_TEP/isambard_1e7/nue_<label>_ladder/tep_ladder_nue_<label>.json (if present; 1e7 walkers)
Acceptance (c) (spec): every peak time within max(0.01, TE) of its target and every ratio within TE
(|z| <= 2 with the bias included), TE = |bias| + 2 SE; reported in the design-dt convention (value at
dt = 1e-3, bias = Q(dt) - R1) and in the continuum convention (R1, bias = R2 - R1).

Run from code/:  python3 fb_v2_nue_summary.py [--no-figure]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEP = HERE.parent / "artifacts" / "data" / "exact_m_fixed_budget" / "V2_TEP"
FIG = HERE.parent / "artifacts" / "figures"
QUEUE = TEP / "_designs" / "nue" / "QUEUE.txt"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _arr(x):
    return np.asarray(x, float)


def sample_row(d: dict) -> dict:
    """compact acceptance metrics of one analysed sample (ladder or bootstrap-analysed independent run)."""
    tc = d["target_checks"]
    pk, rt = tc["peaks"], tc["ratios"]
    R = d["richardson"]
    m = len(pk["target"])
    lv1p, lv1r = R["peaks"]["levels"]["1"], R["ratios"]["levels"]["1"]
    out = {"pass_design_dt": bool(pk["design_dt"]["all_pass"] and rt["design_dt"]["all_pass"]),
           "pass_continuum": bool(pk["continuum"]["all_pass"] and rt["continuum"]["all_pass"]),
           "peaks_pass_design_dt": pk["design_dt"]["pass"], "peaks_pass_continuum": pk["continuum"]["pass"],
           "ratios_pass_design_dt": rt["design_dt"]["pass"], "ratios_pass_continuum": rt["continuum"]["pass"],
           "peak_dev_design_dt": _arr(pk["design_dt"]["deviation"]).tolist(),
           "peak_allowed_design_dt": _arr(pk["design_dt"]["allowed"]).tolist(),
           "peak_dev_continuum": _arr(pk["continuum"]["deviation"]).tolist(),
           "peak_allowed_continuum": _arr(pk["continuum"]["allowed"]).tolist(),
           "peak_TE_design_dt": _arr(lv1p["total_error"])[:m].tolist(),
           "peak_TE_continuum": _arr(R["peaks"]["total_error_continuum"])[:m].tolist(),
           "peak_se_dt": _arr(lv1p["se"])[:m].tolist(),
           "peak_bias_dt": _arr(lv1p["bias"])[:m].tolist(), "peak_bias_se_dt": _arr(lv1p["bias_se"])[:m].tolist(),
           "ratio_dev_design_dt": _arr(rt["design_dt"]["deviation"]).tolist(),
           "ratio_TE_design_dt": _arr(rt["design_dt"]["allowed"]).tolist(),
           "ratio_dev_continuum": _arr(rt["continuum"]["deviation"]).tolist(),
           "ratio_TE_continuum": _arr(rt["continuum"]["allowed"]).tolist(),
           "ratio_bias_dt": _arr(lv1r["bias"])[:m].tolist(), "ratio_bias_se_dt": _arr(lv1r["bias_se"])[:m].tolist(),
           "ratio_z_no_bias_dt": rt["z_design_dt_no_bias"],
           "exactly_m_modes_all_levels": d.get("exactly_m_modes_all_levels"),
           "protocol_P_count_by_level": d.get("protocol_P_count_by_level"),
           "protocol_P_expected_in_window": d.get("protocol_P_expected_in_window"),
           "protocol_P_ok_all_levels": d.get("protocol_P_count_equals_expected_all_levels"),
           "walkers_per_level": d["levels"]["1"]["walkers"], "tag": d["levels"]["1"].get("tag")}
    out["max_abs_peak_dev_design_dt"] = float(np.max(np.abs(out["peak_dev_design_dt"])))
    out["max_abs_peak_dev_continuum"] = float(np.max(np.abs(out["peak_dev_continuum"])))
    out["max_peak_TE_design_dt"] = float(np.max(out["peak_TE_design_dt"]))
    out["max_peak_TE_continuum"] = float(np.max(out["peak_TE_continuum"]))
    out["max_ratio_dev_over_TE_design_dt"] = float(np.max(np.abs(_arr(out["ratio_dev_design_dt"])
                                                                / _arr(out["ratio_TE_design_dt"]))))
    out["max_ratio_dev_over_TE_continuum"] = float(np.max(np.abs(_arr(out["ratio_dev_continuum"])
                                                                / _arr(out["ratio_TE_continuum"]))))
    out["max_abs_ratio_z_no_bias_dt"] = float(np.max(np.abs(out["ratio_z_no_bias_dt"])))
    out["max_abs_peak_bias_dt"] = float(np.max(np.abs(out["peak_bias_dt"])))
    out["max_abs_ratio_bias_dt"] = float(np.max(np.abs(out["ratio_bias_dt"])))
    fk = d.get("fk_vs_dk_at_design_dt") or {}
    out["fk_vs_dk_max_abs_z"] = {q: float(np.max(np.abs(v["z"]))) for q, v in fk.items()}
    out["fk_vs_dk_z"] = {q: list(v["z"]) for q, v in fk.items()}
    # first-order time-step law of the production step (main text Sec. 4.8: peak times early by
    # (dt/2)(1 + gamma t), gamma = 1): expected Q(dt) - Q(0) at the measured peak
    dt = float(d["levels"]["1"]["dt"])
    out["peak_bias_dt_expected_first_order"] = (-(dt / 2) * (1.0 + _arr(lv1p["value"])[:m])).tolist()
    return out


def pooled_row(prim: dict, rep: dict) -> dict:
    """Pooled estimate from both samples (4 x 10^6 independent walkers): the design-step value Q(dt) is the
    inverse-variance mean of four independent estimates -- ladder level 1, independent level 1, and the
    independent levels 2 and 4 shifted to dt by the ladder's paired differences (Q1 - Q2, Q1 - Q4); the
    time-step bias Q(dt) - R1 and R2 - R1 come from the ladder (paired, precise).  Then
        design dt:  TE = |bias_1| + 2 SE(pooled Q(dt));
        continuum:  Q0 = pooled Q(dt) - bias_1, SE0 = sqrt(SE(pooled)^2 + SE(bias_1)^2), TE = |R2 - R1| + 2 SE0.
    chi2_het: heterogeneity of the four estimates about their mean (3 dof per component)."""
    out = {}
    for q in ("peaks", "ratios"):
        A, Bq = prim["richardson"][q], rep["richardson"][q]
        la, lb = A["levels"], Bq["levels"]
        m = len(prim["target_checks"][q]["target"])
        v1a, s1a = _arr(la["1"]["value"])[:m], _arr(la["1"]["se"])[:m]
        d12, sd12 = _arr(A["d12"])[:m], _arr(A["d12_se"])[:m]
        d24, sd24 = _arr(A["d24"])[:m], _arr(A["d24_se"])[:m]
        E = [v1a, _arr(lb["1"]["value"])[:m], _arr(lb["2"]["value"])[:m] + d12, _arr(lb["4"]["value"])[:m] + d12 + d24]
        S = [s1a, _arr(lb["1"]["se"])[:m], np.sqrt(_arr(lb["2"]["se"])[:m] ** 2 + sd12 ** 2),
             np.sqrt(_arr(lb["4"]["se"])[:m] ** 2 + sd12 ** 2 + sd24 ** 2)]
        W = [1 / x ** 2 for x in S]
        Q = sum(w * e for w, e in zip(W, E)) / sum(W)
        SQ = 1 / np.sqrt(sum(W))
        chi2 = sum(w * (e - Q) ** 2 for w, e in zip(W, E))
        b1, sb1 = _arr(la["1"]["bias"])[:m], _arr(la["1"]["bias_se"])[:m]
        bc = _arr(A["bias_continuum"])[:m]
        Q0 = Q - b1
        S0 = np.sqrt(SQ ** 2 + sb1 ** 2)
        T = _arr(prim["target_checks"][q]["target"])
        floor = 0.01 if q == "peaks" else 0.0
        te_dt, te_c = np.abs(b1) + 2 * SQ, np.abs(bc) + 2 * S0
        out[q] = {"value_dt": Q.tolist(), "se_dt": SQ.tolist(), "bias_dt": b1.tolist(), "TE_dt": te_dt.tolist(),
                  "dev_dt": (Q - T).tolist(), "allowed_dt": np.maximum(floor, te_dt).tolist(),
                  "pass_dt": (np.abs(Q - T) <= np.maximum(floor, te_dt)).tolist(),
                  "value_c": Q0.tolist(), "se_c": S0.tolist(), "TE_c": te_c.tolist(), "dev_c": (Q0 - T).tolist(),
                  "allowed_c": np.maximum(floor, te_c).tolist(),
                  "pass_c": (np.abs(Q0 - T) <= np.maximum(floor, te_c)).tolist(),
                  "z_dt_no_bias": ((Q - T) / SQ).tolist(), "chi2_het_3dof": chi2.tolist()}
    fp = prim["design"].get("fk_prediction")
    if fp:
        for q in ("peaks", "ratios"):
            if q in fp:
                n = len(fp[q])
                sf = _arr((fp.get("se") or {}).get(q, np.zeros(n)))
                out[q]["z_vs_fk_oos"] = ((_arr(out[q]["value_dt"])[:n] - _arr(fp[q])) /
                                         np.sqrt(_arr(out[q]["se_dt"])[:n] ** 2 + sf ** 2)).tolist()
                out[q]["dev_vs_fk_oos"] = (_arr(out[q]["value_dt"])[:n] - _arr(fp[q])).tolist()
    out["pass_design_dt"] = bool(all(all(out[q]["pass_dt"]) for q in ("peaks", "ratios")))
    out["pass_continuum"] = bool(all(all(out[q]["pass_c"]) for q in ("peaks", "ratios")))
    out["max_abs_peak_dev_dt"] = float(np.max(np.abs(out["peaks"]["dev_dt"])))
    out["max_abs_peak_dev_c"] = float(np.max(np.abs(out["peaks"]["dev_c"])))
    out["max_peak_TE_dt"] = float(np.max(out["peaks"]["TE_dt"]))
    out["max_peak_TE_c"] = float(np.max(out["peaks"]["TE_c"]))
    out["max_ratio_dev_over_TE_dt"] = float(np.max(np.abs(_arr(out["ratios"]["dev_dt"]) / _arr(out["ratios"]["TE_dt"]))))
    out["max_ratio_dev_over_TE_c"] = float(np.max(np.abs(_arr(out["ratios"]["dev_c"]) / _arr(out["ratios"]["TE_c"]))))
    out["max_abs_ratio_z_dt"] = float(np.max(np.abs(out["ratios"]["z_dt_no_bias"])))
    out["max_abs_peak_z_dt"] = float(np.max(np.abs(out["peaks"]["z_dt_no_bias"])))
    return out


def pooled_aggregate(rows: dict) -> dict:
    P = [r["pooled"] for r in rows.values() if "pooled" in r]
    if not P:
        return {}
    def zs(q, k="z_dt_no_bias", first=None):
        out = []
        for r in rows.values():
            if "pooled" in r:
                z = r["pooled"][q][k]
                out += z[:(len(z) - 1 if first == "m-1" else len(z))]
        return _arr(out)
    zr, zp = zs("ratios", first="m-1"), zs("peaks")
    het = np.concatenate([zs("peaks", "chi2_het_3dof"), zs("ratios", "chi2_het_3dof", "m-1")])
    from math import erf, sqrt
    p2 = 1 - erf(2 / sqrt(2))
    n_chk_r = sum(len(r["pooled"]["ratios"]["pass_dt"]) for r in rows.values() if "pooled" in r)
    n_fail_r = sum(int(np.sum(~np.asarray(r["pooled"]["ratios"]["pass_dt"]))) for r in rows.values() if "pooled" in r)
    n_chk_p = sum(len(r["pooled"]["peaks"]["pass_dt"]) for r in rows.values() if "pooled" in r)
    n_fail_p = sum(int(np.sum(~np.asarray(r["pooled"]["peaks"]["pass_dt"]))) for r in rows.values() if "pooled" in r)
    n_fail_rc = sum(int(np.sum(~np.asarray(r["pooled"]["ratios"]["pass_c"]))) for r in rows.values() if "pooled" in r)
    n_fail_pc = sum(int(np.sum(~np.asarray(r["pooled"]["peaks"]["pass_c"]))) for r in rows.values() if "pooled" in r)
    return {"n_designs": len(P), "n_pass_design_dt": sum(x["pass_design_dt"] for x in P),
            "n_pass_continuum": sum(x["pass_continuum"] for x in P),
            "failures_design_dt": [lab for lab, r in rows.items() if "pooled" in r and not r["pooled"]["pass_design_dt"]],
            "failures_continuum": [lab for lab, r in rows.items() if "pooled" in r and not r["pooled"]["pass_continuum"]],
            "peak_checks": {"n": n_chk_p, "n_fail_dt": n_fail_p, "n_fail_continuum": n_fail_pc},
            "ratio_checks": {"n": n_chk_r, "n_fail_dt": n_fail_r, "n_fail_continuum": n_fail_rc,
                             "expected_fail_if_exact_at_2sigma": float(n_chk_r * p2)},
            "max_abs_peak_dev_dt": max(x["max_abs_peak_dev_dt"] for x in P),
            "max_abs_peak_dev_c": max(x["max_abs_peak_dev_c"] for x in P),
            "max_peak_TE_dt": max(x["max_peak_TE_dt"] for x in P),
            "max_ratio_dev_over_TE_dt": max(x["max_ratio_dev_over_TE_dt"] for x in P),
            "max_ratio_dev_over_TE_c": max(x["max_ratio_dev_over_TE_c"] for x in P),
            "ratio_z_dt_pooled_first_m_minus_1": {"n": int(zr.size), "mean_z2": float(np.mean(zr ** 2)),
                                                  "max_abs_z": float(np.max(np.abs(zr))),
                                                  "frac_abs_z_gt_2": float(np.mean(np.abs(zr) > 2))},
            "peak_z_dt_pooled": {"n": int(zp.size), "mean_z2": float(np.mean(zp ** 2)),
                                 "max_abs_z": float(np.max(np.abs(zp))), "frac_abs_z_gt_2": float(np.mean(np.abs(zp) > 2))},
            "z_vs_fk_oos": {q: (lambda z: {"n": int(z.size), "mean_z2": float(np.mean(z ** 2)),
                                           "max_abs_z": float(np.max(np.abs(z))),
                                           "frac_abs_z_gt_2": float(np.mean(np.abs(z) > 2))} if z.size else None)(
                                _arr([x for r in rows.values() if "pooled" in r
                                      for x in r["pooled"][q].get("z_vs_fk_oos", [])])) for q in ("peaks", "ratios")},
            "heterogeneity_chi2_3dof": {"n": int(het.size), "mean_over_3": float(np.mean(het) / 3),
                                        "max": float(np.max(het))}}


def bias_law_check(rows: dict, smp: str = "primary") -> dict:
    """pooled comparison of the measured level-1 peak-time bias (paired ladder) with -(dt/2)(1 + t)."""
    meas, exp_, se = [], [], []
    for r in rows.values():
        if smp in r:
            P = r[smp]
            meas += P["peak_bias_dt"]
            exp_ += P["peak_bias_dt_expected_first_order"]
            se += P["peak_bias_se_dt"]
    if not meas:
        return {}
    meas, exp_, se = _arr(meas), _arr(exp_), _arr(se)
    z = (meas - exp_) / se
    w = 1 / se ** 2
    slope = float(np.sum(w * meas * exp_) / np.sum(w * exp_ ** 2))
    return {"sample": smp, "n_peaks": int(meas.size), "weighted_slope_measured_on_expected": slope,
            "chi2": float(np.sum(z ** 2)), "dof": int(meas.size), "max_abs_z": float(np.max(np.abs(z))),
            "median_bias_se": float(np.median(se)),
            "note": "expected Q(dt) - Q(0) = -(dt/2)(1 + gamma t) per peak (first order, gamma = 1)"}


def replicate_consistency(pairs: list) -> dict:
    """ladder vs independent-levels run, same design and level: z = (Q_lad - Q_ind)/sqrt(se_lad^2 + se_ind^2)
    for peaks and ratios at every level (tests the bootstrap SEs; both samples estimate the same law)."""
    zs = {"peaks": [], "ratios": []}
    for prim, rep in pairs:
        for q in zs:
            for L in ("1", "2", "4"):
                a, b = prim["richardson"][q]["levels"][L], rep["richardson"][q]["levels"][L]
                z = (_arr(a["value"]) - _arr(b["value"])) / np.sqrt(_arr(a["se"]) ** 2 + _arr(b["se"]) ** 2)
                zs[q] += z[np.isfinite(z)].tolist()
    out = {}
    for q, z in zs.items():
        z = _arr(z)
        if z.size:
            out[q] = {"n": int(z.size), "chi2_over_n": float(np.mean(z ** 2)), "max_abs_z": float(np.max(np.abs(z))),
                      "frac_abs_z_gt_2": float(np.mean(np.abs(z) > 2))}
    return out


def collect() -> dict:
    labels = [Path(x).stem for x in QUEUE.read_text().split() if x.strip()]
    rows, missing, pairs = {}, [], []
    for lab in labels:
        prim = _load(TEP / f"{lab}_ladder" / f"tep_ladder_{lab}.json")
        rep = _load(TEP / lab / f"tep_boot_{lab}.json")
        if prim and rep:
            pairs.append((prim, rep))
        har = _load(TEP / lab / f"tep_{lab}.json")
        isb = _load(TEP / "isambard_1e7" / f"{lab}_ladder" / f"tep_ladder_{lab}.json")
        if prim is None and rep is None:
            missing.append(lab)
            continue
        dsg = (prim or rep)["design"]
        row = {"design_label": dsg.get("design_label", lab[4:]), "m": len(dsg["w"]), "eps": dsg["eps"],
               "B": dsg["B"], "targets": dsg.get("targets"), "tmax": dsg.get("tmax", 4.0),
               "design_source": dsg.get("design_source"),
               "sources": {"primary": f"V2_TEP/{lab}_ladder/tep_ladder_{lab}.json" if prim else None,
                           "replicate": f"V2_TEP/{lab}/tep_boot_{lab}.json" if rep else None,
                           "harness_jackknife": f"V2_TEP/{lab}/tep_{lab}.json" if har else None,
                           "isambard_1e7": f"V2_TEP/isambard_1e7/{lab}_ladder/tep_ladder_{lab}.json" if isb else None}}
        if prim:
            row["primary"] = sample_row(prim)
        if prim and rep:
            row["pooled"] = pooled_row(prim, rep)
        if rep:
            row["replicate"] = sample_row(rep)
        if isb:
            row["isambard_1e7"] = sample_row(isb)
        if har and "target_checks" in har:
            row["harness_jackknife_pass"] = {"design_dt": all(c["design_dt"]["all_pass"] for c in har["target_checks"].values()),
                                             "continuum": all(c["continuum"]["all_pass"] for c in har["target_checks"].values())}
        rows[lab] = row
    agg = {}
    for smp in ("primary", "replicate", "isambard_1e7"):
        rs = [r[smp] for r in rows.values() if smp in r]
        if not rs:
            continue
        agg[smp] = {"n_designs": len(rs),
                    "n_pass_design_dt": sum(r["pass_design_dt"] for r in rs),
                    "n_pass_continuum": sum(r["pass_continuum"] for r in rs),
                    "n_exactly_m_modes_all_levels": sum(bool(r["exactly_m_modes_all_levels"]) for r in rs),
                    "n_protocol_P_count_ok_all_levels": sum(bool(r["protocol_P_ok_all_levels"]) for r in rs),
                    "n_with_protocol_P": sum(r["protocol_P_ok_all_levels"] is not None for r in rs),
                    "max_abs_peak_dev_design_dt": max(r["max_abs_peak_dev_design_dt"] for r in rs),
                    "max_abs_peak_dev_continuum": max(r["max_abs_peak_dev_continuum"] for r in rs),
                    "max_peak_TE_design_dt": max(r["max_peak_TE_design_dt"] for r in rs),
                    "n_peaks_TE_above_0.01_design_dt": sum(int(np.sum(_arr(r["peak_TE_design_dt"]) > 0.01)) for r in rs),
                    "n_peaks_total": sum(len(r["peak_TE_design_dt"]) for r in rs),
                    "max_ratio_dev_over_TE_design_dt": max(r["max_ratio_dev_over_TE_design_dt"] for r in rs),
                    "max_ratio_dev_over_TE_continuum": max(r["max_ratio_dev_over_TE_continuum"] for r in rs),
                    "max_abs_ratio_z_no_bias_dt": max(r["max_abs_ratio_z_no_bias_dt"] for r in rs),
                    "max_abs_peak_bias_dt": max(r["max_abs_peak_bias_dt"] for r in rs),
                    "max_abs_ratio_bias_dt": max(r["max_abs_ratio_bias_dt"] for r in rs),
                    "fk_vs_dk_max_abs_z_peaks": max((r["fk_vs_dk_max_abs_z"]["peaks"] for r in rs
                                                     if "peaks" in r["fk_vs_dk_max_abs_z"]), default=None),
                    "fk_vs_dk_max_abs_z_ratios": max((r["fk_vs_dk_max_abs_z"]["ratios"] for r in rs
                                                      if "ratios" in r["fk_vs_dk_max_abs_z"]), default=None),
                    "n_designs_with_fk_prediction": sum(bool(r["fk_vs_dk_max_abs_z"]) for r in rs),
                    "fk_vs_dk_pooled": {q: (lambda z: {"n": int(z.size), "mean_z2": float(np.mean(z ** 2)),
                                                       "frac_abs_z_gt_2": float(np.mean(np.abs(z) > 2)),
                                                       "n_abs_z_gt_3": int(np.sum(np.abs(z) > 3))} if z.size else None)(
                                            _arr([x for r in rs for x in r["fk_vs_dk_z"].get(q, [])]))
                                        for q in ("peaks", "ratios")},
                    "failures_design_dt": [lab for lab, r in rows.items() if smp in r and not r[smp]["pass_design_dt"]],
                    "failures_continuum": [lab for lab, r in rows.items() if smp in r and not r[smp]["pass_continuum"]]}
    agg["bias_law_check_primary"] = bias_law_check(rows, "primary")
    agg["pooled"] = pooled_aggregate(rows)
    # resolved time-step bias of the ladder (paired): sign, size, observed order
    mb = _arr([x for r in rows.values() if "primary" in r for x in r["primary"]["peak_bias_dt"]])
    mr = _arr([x for r in rows.values() if "primary" in r for x in r["primary"]["ratio_bias_dt"]])
    rho = []
    for lab in rows:
        d = _load(TEP / f"{lab}_ladder" / f"tep_ladder_{lab}.json")
        if d:
            R = d["richardson"]["peaks"]
            rho += [rr for dd, sd, rr in zip(R["d24"], R["d24_se"], R["order_ratio"]) if abs(dd) > 3 * sd]
    if mb.size:
        agg["time_step_bias_ladder"] = {
            "peak_bias_dt_min": float(mb.min()), "peak_bias_dt_max": float(mb.max()),
            "peak_bias_frac_negative": float(np.mean(mb < 0)), "ratio_bias_dt_max_abs": float(np.max(np.abs(mr))),
            "order_ratio_peaks_resolved": {"n": len(rho), "median": float(np.median(rho)) if rho else None,
                                           "q25": float(np.percentile(rho, 25)) if rho else None,
                                           "q75": float(np.percentile(rho, 75)) if rho else None,
                                           "selection": "|Q(dt/2) - Q(dt/4)| > 3 SE (paired)"}}
    grp = {}
    for r in rows.values():
        if "pooled" in r and "z_vs_fk_oos" in r["pooled"]["peaks"]:
            g = grp.setdefault(f"eps{r['eps']:g}", {"z": [], "dev": []})
            g["z"] += r["pooled"]["peaks"]["z_vs_fk_oos"]
            g["dev"] += r["pooled"]["peaks"]["dev_vs_fk_oos"]
    agg["pooled"]["peaks_vs_fk_oos_by_eps"] = {k: {"n": len(v["z"]), "mean_z": float(np.mean(v["z"])),
                                                   "mean_z2": float(np.mean(_arr(v["z"]) ** 2)),
                                                   "max_abs_dev": float(np.max(np.abs(v["dev"])))}
                                               for k, v in grp.items()}
    zr = agg["pooled"].get("ratio_z_dt_pooled_first_m_minus_1")
    if zr:
        from math import erfc, sqrt
        agg["pooled"]["ratio_max_abs_z_bonferroni_p"] = float(min(1.0, zr["n"] * erfc(zr["max_abs_z"] / sqrt(2))))
    agg["replicate_consistency_ladder_vs_independent"] = replicate_consistency(pairs)
    groups = {}
    for lab, r in rows.items():
        g = ("horizon" if "_hz_" in lab else "showcase" if "showcase" in lab else "nocap" if "nocap" in lab
             else "demo")
        r["group"] = g
        groups.setdefault(g, []).append(lab)
    agg["groups"] = {g: {"n": len(v), "labels": v,
                         "n_pass_design_dt_primary": sum(rows[x]["primary"]["pass_design_dt"] for x in v if "primary" in rows[x]),
                         "n_pass_continuum_primary": sum(rows[x]["primary"]["pass_continuum"] for x in v if "primary" in rows[x]),
                         "n_primary": sum("primary" in rows[x] for x in v)} for g, v in groups.items()}
    return {"item": "V2 NU-E: out-of-sample direct-kill validation of the V2 designs under the TEP",
            "driver": "code/fb_v2_nue_summary.py", "n_designs_expected": len(labels), "missing": missing,
            "acceptance_rule": "peaks |Q - T| <= max(0.01, TE); ratios |Q - r| <= TE; TE = |bias| + 2 SE "
                               "(design-dt: Q = Q(1e-3), bias = Q(dt) - R1; continuum: Q = R1, bias = R2 - R1)",
            "aggregate": agg, "designs": rows}


def figure(S: dict) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = [(lab, r) for lab, r in S["designs"].items() if "pooled" in r]
    if not rows:
        return None
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.hashsalt": "nue"})
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    col = {3: "#2ca02c", 4: "#1f77b4", 5: "#d62728"}
    ticks, labs = [], []
    for x, (lab, r) in enumerate(rows):
        P, m = r["pooled"], r["m"]
        xs = x + (np.arange(m) - (m - 1) / 2) * 0.13
        dev, te = _arr(P["peaks"]["dev_dt"]), _arr(P["peaks"]["TE_dt"])
        ax[0].errorbar(xs, dev, yerr=te, fmt="o", ms=1.8, lw=0.6, color=col.get(m, "k"), capsize=0)
        rd, rte = _arr(P["ratios"]["dev_dt"]), _arr(P["ratios"]["TE_dt"])
        ax[1].plot(xs, rd / rte, "o", ms=1.8, color=col.get(m, "k"))
        prof = next((k[0] for k in ("rising", "falling", "equal") if k in lab), "")
        tag = {"horizon": ("tmax" if "zbar" in lab else "cap"), "showcase": "show", "nocap": f"t{int(r['tmax'])}",
               "demo": prof}[r["group"]]
        ticks.append(x)
        labs.append(f"{m},{r['eps']:g},{r['B']:g},{tag}")
    ax[0].axhspan(-0.01, 0.01, color="0.9", zorder=0)
    ax[1].axhspan(-1, 1, color="0.9", zorder=0)
    for a in ax:
        a.axhline(0, color="0.5", lw=0.5)
    ax[0].set_ylabel("peak time $-$ target")
    ax[1].set_ylabel("(ratio $-$ target) / TE")
    ax[0].set_title("(a) peak times at $\\Delta t=10^{-3}$ (pooled direct kill, $4\\times10^6$ walkers); bars: total "
                    "error; shaded: $\\pm0.01$", fontsize=7, loc="left")
    ax[1].set_title("(b) conditional mass ratios; shaded: within the total error", fontsize=7, loc="left")
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labs, rotation=90, fontsize=5)
    ax[0].legend(handles=[Line2D([], [], color=c, marker="o", ls="", ms=3, label=f"$m={k}$") for k, c in col.items()],
                 fontsize=6, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fb_v2_tep_validation.pdf"
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


def _sci(x: float) -> str:
    """fixed-point with five decimals (all tabulated deviations and errors are below 0.1)."""
    return f"{x:.5f}" if abs(x) < 0.1 else f"{x:.4f}"


def tex_rows(S: dict) -> str:
    """table body: one row per design (pooled two-sample metrics; pass flags pooled | ladder | independent)."""
    lines = []
    for lab, r in S["designs"].items():
        if "pooled" not in r:
            continue
        P, L1, R1 = r["pooled"], r["primary"], r.get("replicate")
        prof = next((k for k in ("rising", "falling", "equal") if k in lab), "")
        kind = {"horizon": "hz", "showcase": "show", "nocap": "tmax" + str(int(r["tmax"])), "demo": prof}[r["group"]]
        if r["group"] == "horizon":
            kind = "$t_{\\max}$" if "zbar" in lab else "cap"
        ok = lambda b: "\\checkmark" if b else "$\\times$"                              # noqa: E731
        lines.append(f"% src: artifacts/data/exact_m_fixed_budget/V2_TEP/nue_summary.json#designs.{lab}.{{pooled,primary,replicate}}")
        lines.append(" & ".join([str(r["m"]), f"{r['eps']:g}", f"{r['B']:g}", kind,
                                 _sci(P["max_abs_peak_dev_dt"]), _sci(P["max_peak_TE_dt"]),
                                 _sci(P["max_abs_peak_dev_c"]),
                                 f"{P['max_ratio_dev_over_TE_dt']:.2f}", f"{P['max_ratio_dev_over_TE_c']:.2f}",
                                 ok(P["pass_design_dt"]) + ok(P["pass_continuum"]),
                                 ok(L1["pass_design_dt"]) + ok(L1["pass_continuum"]),
                                 (ok(R1["pass_design_dt"]) + ok(R1["pass_continuum"])) if R1 else "--"]) + "\\\\")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--tex", default=None, help="write the table body to this file")
    a = ap.parse_args(argv)
    S = collect()
    out = TEP / "nue_summary.json"
    out.write_text(json.dumps(S, indent=1))
    if a.tex:
        Path(a.tex).write_text(tex_rows(S) + "\n")
    print(json.dumps(S["aggregate"], indent=1))
    print("missing", S["missing"])
    if not a.no_figure:
        print("figure", figure(S))
    return 0


if __name__ == "__main__":
    sys.exit(main())
