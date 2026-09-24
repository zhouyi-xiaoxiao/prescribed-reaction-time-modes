#!/usr/bin/env python3
"""Analyst report for HPC_N5full and HPC_N3full (read-only on the fetched JSON).

Reads the two summaries fetched from Isambard 3,
    artifacts/data/exact_m_fixed_budget/HPC_N5full/hpc_n5full.json   (jobs 5768127 + 5768647)
    artifacts/data/exact_m_fixed_budget/HPC_N3full/hpc_n3full.json   (jobs 5768126 + 5768647)
plus the local reference results N5/n5_dt_ladder.json and N3/n3_summary.json, and
writes
    HPC_N5full/hpc_n5full_assessment.json   (acceptance re-evaluation, EM vs exact-OU split)
    HPC_N3full/hpc_n3full_assessment.json   (contact / field claims vs local N3)
    artifacts/figures/fb_hpc_n5full.{pdf,png}, fb_hpc_n3full.{pdf,png}   (core.save_figure)
    manuscript/cnsns_submission/results_snippets/HPC_N5full.tex, HPC_N3full.tex
No simulation, no randomness: every output is a deterministic function of the JSON inputs.

Two derived diagnostics are computed here (not on the cluster); both are exact
algebra on fetched numbers:
* time-label correction of the exact-OU chain.  Every scheme charges the
  exposure of a step (t_{k-1}, t_k] with the field at the END point t_k, so its
  kill-time law is the continuum law advanced by Delta t/2 to first order.
  Relabelling by +Delta t/2 gives: peak time + Delta t/2; basin mass on
  (a, b]  ->  M - (Delta t/2) [f(b) - f(a)], with the density f at a basin edge
  taken as the mean of the two adjacent 0.02-bin masses / 0.02 (bin_mass of
  the fetched JSON); relative prominence is shift invariant (unchanged).
* Euler--Maruyama clock.  The EM OU chain contracts by (1 - gamma dt) per step,
  i.e. at rate -ln(1 - gamma dt)/dt = gamma (1 + gamma dt/2 + ...), so its mean
  path reaches a stripe earlier by gamma t dt/2; with the end-point exposure
  this gives the EM peak offset (dt/2)(1 + gamma t) (gamma = 1 here).

Usage (from code/):  python3 fb_hpc_n5n3_report.py [all|derive|figures|snippets]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
DATA = REPORT / "artifacts" / "data" / "exact_m_fixed_budget"
SNIP = REPORT / "manuscript" / "cnsns_submission" / "results_snippets"
sys.path.insert(0, str(HERE.parent))

P_N5 = DATA / "HPC_N5full" / "hpc_n5full.json"
P_N3 = DATA / "HPC_N3full" / "hpc_n3full.json"
P_N5L = DATA / "N5" / "n5_dt_ladder.json"
P_N3L = DATA / "N3" / "n3_summary.json"
O_N5 = DATA / "HPC_N5full" / "hpc_n5full_assessment.json"
O_N3 = DATA / "HPC_N3full" / "hpc_n3full_assessment.json"

DT_PROD = 1e-3
BIN = 0.02
ORDER5 = ["m3_anchor", "m3_threshold", "m3_phase_boundary", "m5_anchor"]
LABEL5 = {"m3_anchor": r"$(3,0.1,1)$", "m3_threshold": r"$(3,0.1,3.57)$",
          "m3_phase_boundary": r"$(3,0.175,0.5)$", "m5_anchor": r"$(5,0.1,1)$"}
LABEL5_2L = {"m3_anchor": "(3, 0.1,\n1)", "m3_threshold": "(3, 0.1,\n3.57)",
             "m3_phase_boundary": "(3, 0.175,\n0.5)", "m5_anchor": "(5, 0.1,\n1)"}


def rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(REPORT))
    except ValueError:
        return str(p)


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


# ----------------------------------------------------------------------------
# N5
# ----------------------------------------------------------------------------

def _budgets(D5: dict) -> dict:
    out = {}
    for ck, c in D5["cells"].items():
        for bk, b in c["budgets"].items():
            out[bk] = (ck, b)
    return out


def _edge_density(bm: np.ndarray, i: int) -> float:
    n = bm.size
    if 0 < i < n:
        return 0.5 * (bm[i - 1] + bm[i]) / BIN
    return bm[min(max(i, 0), n - 1)] / BIN


def derive_n5() -> dict:
    D = load(P_N5)
    DL = load(P_N5L)
    B5 = _budgets(D)
    rows = []
    for lab in ORDER5:
        ck, b = B5[lab]
        F = b["functionals"]
        m = int(b["m"])
        ci = b["cuts_idx"]
        for fam in ("basin_masses", "peak_times_smoothed", "late_rel_prominence_smoothed"):
            keys = [f"{fam}[{j}]" for j in range(m)] if fam != "late_rel_prominence_smoothed" else [fam]
            for j, key in enumerate(keys):
                f = F[key]
                v = f["values"]
                L = f["exb_richardson2"]
                hw = f["prod_value_ci95_halfwidth"]
                em = v["em_r2"] - L
                ex = v["ex_r2"] - L
                if fam == "basin_masses":
                    bm = np.asarray(b["bin_mass"]["ex_r2"], float)
                    corr = -(DT_PROD / 2) * (_edge_density(bm, ci[j + 1]) - _edge_density(bm, ci[j]))
                    exl = float(ex + corr)
                elif fam == "peak_times_smoothed":
                    exl = ex + DT_PROD / 2
                else:
                    exl = ex
                rows.append({
                    "budget": lab, "cell": ck, "functional": key, "family": fam,
                    "limit_exb_R2": L, "ci95_halfwidth_dt1e-3": hw,
                    "em_r2_minus_limit": em, "ex_r2_minus_limit": ex,
                    "ex_r2_relabelled_minus_limit": exl,
                    "plan_D_em_R1_minus_em_r2": f["prod_bias_D"],
                    "plan_D_se": f["prod_bias_D_se"],
                    "em_R2_minus_exb_R2": f.get("em_vs_exb_richardson2_diff"),
                    "em_R2_vs_exb_R2_z": f.get("em_vs_exb_richardson2_z"),
                    "em_order_ratio": f.get("em_order_ratio"),
                    "ex_order_ratio": f.get("ex_order_ratio"),
                    "pass_em": bool(abs(em) < hw), "pass_ex": bool(abs(ex) < hw), "pass_ex_relabelled": bool(abs(exl) < hw),
                    "pass_plan_em_R1": bool(abs(f["prod_bias_D"]) < hw),
                })
    fams = ("basin_masses", "peak_times_smoothed", "late_rel_prominence_smoothed")
    counts = {}
    for fam in fams:
        rr = [r for r in rows if r["family"] == fam]
        counts[fam] = {"n": len(rr)}
        for s in ("em", "ex", "ex_relabelled", "plan_em_R1"):
            counts[fam][f"n_pass_{s}"] = sum(r[f"pass_{s}"] for r in rr)
        for s, k in (("em", "em_r2_minus_limit"), ("ex", "ex_r2_minus_limit"),
                     ("ex_relabelled", "ex_r2_relabelled_minus_limit")):
            counts[fam][f"max_abs_over_hw_{s}"] = max(abs(r[k]) / r["ci95_halfwidth_dt1e-3"] for r in rr)
            counts[fam][f"max_abs_{s}"] = max(abs(r[k]) for r in rr)
        zz = [abs(r["em_R2_vs_exb_R2_z"]) for r in rr if r["em_R2_vs_exb_R2_z"] is not None]
        dd = [abs(r["em_R2_minus_exb_R2"]) for r in rr if r["em_R2_minus_exb_R2"] is not None]
        counts[fam]["max_abs_em_R2_minus_exb_R2"] = max(dd)
        counts[fam]["max_abs_z_em_R2_vs_exb_R2"] = max(zz)
    for fam in fams:
        ses = []
        for lab in ORDER5:
            ck, b = B5[lab]
            for k, f in b["functionals"].items():
                if (k.startswith(fam + "[") or k == fam) and "prod_bias_vs_exb_R2_se" in f:
                    ses.append((f["prod_bias_vs_exb_R2_se"], lab, k))
        mx = max(ses)
        counts[fam]["max_paired_se_prod_bias_vs_exb_R2"] = {"value": mx[0], "budget": mx[1], "functional": mx[2]}
    mode_counts = {}
    for lab in ORDER5:
        ck, b = B5[lab]
        for nn in ("500000", "1000000"):
            vals = b["functionals"][f"protocol_mode_count_N{nn}"]["values"]
            mode_counts[f"{lab}|N{nn}"] = {"m": b["m"], "counts": sorted(set(vals.values())),
                                         "same_all_schemes": len(set(vals.values())) == 1}
    counts["total"] = {k: sum(counts[f][k] for f in fams)
                       for k in ("n", "n_pass_em", "n_pass_ex", "n_pass_ex_relabelled", "n_pass_plan_em_R1")}

    # peak-time decomposition
    pb = D["peak_time_bias_model"]["rows"]
    dec = []
    for r in pb:
        em_minus_ex = r["D_em_vs_exb_R2"] - r["D_ex_r2_vs_exb_R2"]      # EM-specific advance
        t = r["t_peak_exb_R2"]
        dec.append({"cell": r["cell"], "peak": r["peak"], "t_peak": t,
                    "advance_em": r["D_em_vs_exb_R2"], "advance_exactOU": r["D_ex_r2_vs_exb_R2"],
                    "advance_em_minus_exactOU": em_minus_ex,
                    "pred_em_specific_gamma_t_dt2": 0.5 * DT_PROD * t,
                    "em_specific_share": em_minus_ex / r["D_em_vs_exb_R2"]})
    peak_dec = {
        "rows": dec,
        "exactOU_advance_range": [min(x["advance_exactOU"] for x in dec), max(x["advance_exactOU"] for x in dec)],
        "max_abs_exactOU_advance_minus_dt2": max(abs(x["advance_exactOU"] - DT_PROD / 2) for x in dec),
        "max_abs_em_specific_minus_gamma_t_dt2": max(abs(x["advance_em_minus_exactOU"] - x["pred_em_specific_gamma_t_dt2"]) for x in dec),
        "em_advance_range": [min(x["advance_em"] for x in dec), max(x["advance_em"] for x in dec)],
        "em_specific_share_range": [min(x["em_specific_share"] for x in dec), max(x["em_specific_share"] for x in dec)],
        "max_abs_D_em_vs_exb_minus_pred_(cluster)": D["peak_time_bias_model"]["max_abs_D_em_vs_exb_minus_pred"],
        "note": "advance = limit (exb_R2) minus value at dt = 1e-3; positive = early. EM-specific part = EM advance minus exact-OU advance.",
    }

    # prominence: EM vs exact OU
    prom = {}
    for lab in ORDER5:
        r = next(x for x in rows if x["budget"] == lab and x["family"] == "late_rel_prominence_smoothed")
        prom[lab] = {"limit": r["limit_exb_R2"], "hw": r["ci95_halfwidth_dt1e-3"],
                     "em_r2_minus_limit": r["em_r2_minus_limit"], "ex_r2_minus_limit": r["ex_r2_minus_limit"],
                     "share_removed_by_exactOU": 1.0 - r["ex_r2_minus_limit"] / r["em_r2_minus_limit"],
                     "plan_D_em": r["plan_D_em_R1_minus_em_r2"], "plan_D_se": r["plan_D_se"]}
    FL = DL["cells"]["m5_z08_eps0.1"]["budgets"]["m5_anchor"]["functionals"]["late_rel_prominence_smoothed"]
    prom["m5_anchor"]["local_plan_D_em"] = FL["prod_bias_D"]
    prom["m5_anchor"]["local_plan_D_se"] = FL["prod_bias_D_se"]
    prom["m5_anchor"]["local_hw"] = FL["prod_value_ci95_halfwidth"]
    prom["m5_anchor"]["local_ex_r2_minus_exR2"] = FL["values"]["ex_r2"] - FL["ex_richardson2"]
    prom["m5_anchor"]["z_hpc_vs_local_plan_D"] = (
        (prom["m5_anchor"]["plan_D_em"] - FL["prod_bias_D"])
        / math.hypot(prom["m5_anchor"]["plan_D_se"], FL["prod_bias_D_se"]))

    # local peak-time failures, for the record
    lpb = DL["peak_time_bias_model"]
    local_peak = {"n_fail": sum(1 for c in DL["cells"].values() for b in c["budgets"].values()
                                for k, f in b["functionals"].items()
                                if k.startswith("peak_times_smoothed[") and not f["accept_abs_D_lt_ci95_of_dt1e-3_value"]),
                  "D_em_range": [min(r["D_em"] for r in lpb["rows"]), max(r["D_em"] for r in lpb["rows"])]}

    # boundary cell
    bf = D["boundary_flip"]
    ck, bpb = B5["m3_phase_boundary"]
    Fp = bpb["functionals"]["late_rel_prominence_smoothed"]
    boundary = {
        "stored_pair": bf["stored_pair"],
        "resampling": {k: {"p": v["p_mode_count_eq_m"], "wilson95": v["wilson95"], "walkers": v["walkers"],
                           "replicas": v["replicas"]} for k, v in bf["resampling"].items()},
        "p_observed_pattern_3_then_2_at_5e5": bf["p_observed_pattern_3_then_2_at_5e5"],
        "late_rel_prominence_limit": Fp["exb_richardson2"],
        "late_rel_prominence_em_r2": Fp["values"]["em_r2"],
        "late_rel_prominence_em_r1": Fp["values"]["em_r1"],
        "late_rel_prominence_hw": Fp["prod_value_ci95_halfwidth"],
        "em_r1_minus_em_r2": Fp["values"]["em_r1"] - Fp["values"]["em_r2"],
        "em_diff_mid_minus_fine_se": Fp["em_diff_mid_minus_fine_se"],
        "reading": "P(3 modes) at 5e5 walkers is 0.991 at dt = 1e-3 and 0.9885 at dt = 5e-4 (Wilson intervals overlap); "
                   "the stored 3 -> 2 pair is a multinomial sampling event of a cell whose last-mode prominence "
                   "(about 6%) sits just above the 5% floor, not a time-step effect.",
    }

    out = {
        "item": "HPC_N5full assessment (analyst 1)",
        "inputs": {"hpc": rel(P_N5), "local": rel(P_N5L)},
        "limit": "exb_R2 = second-order Richardson limit of the bridge-refined exact-OU ladder (dt = 1.25e-4, 2.5e-4, 5e-4)",
        "criterion": "|f_scheme(dt = 1e-3) - limit| < 95% sampling half-width of the dt = 1e-3 production value (1.96 SE, 3e6 paths)",
        "time_label_correction": "exact-OU values relabelled by +dt/2 (end-point exposure): peak + dt/2; mass - (dt/2)[f(b) - f(a)]; prominence unchanged",
        "rows": rows,
        "counts": counts,
        "protocol_mode_count_by_scheme": mode_counts,
        "peak_time_decomposition": peak_dec,
        "prominence": prom,
        "local_peak_time_failures": local_peak,
        "boundary_cell": boundary,
        "local_N5_max_abs_z_em_r2": {k: v["max_abs_z_em_r2"] for k, v in D["local_N5_comparison"].items()
                                     if isinstance(v, dict) and "max_abs_z_em_r2" in v},
    }
    O_N5.write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------
# N3
# ----------------------------------------------------------------------------

LOCAL_KEY = {"m2_eps0.1": "m2", "m3_eps0.1": "m3"}


def _local_cells(la: dict) -> dict:
    out = {}
    for c in la["cells"]:
        out[(f"g{c['gate']}_a{c['a']}_f{c['field']}", float(c["B"]))] = c
    return out


def derive_n3() -> dict:
    D = load(P_N3)
    DL = load(P_N3L)
    res = {"item": "HPC_N3full assessment (analyst 1)", "inputs": {"hpc": rel(P_N3), "local": rel(P_N3L)},
           "level_used": "R1 (first-order Richardson limit of the exact-OU ladder) unless stated",
           "basins": "window basins [0.5, G valley(s), 3.5] (local N3 / paper convention)", "anchors": {}}
    for an, A in D["anchors"].items():
        la = DL["anchors"][LOCAL_KEY[an]]
        LC = _local_cells(la)
        C = {(c["level"], c["variant"], float(c["B"])): c for c in A["cells"]}
        Bs = [float(x) for x in A["budgets"]]
        V = A["variants"]
        m = int(A["m"])
        # 1. mode counts: level stability and agreement with the local run
        n_cells = len(V) * len(Bs)
        n_level_change = 0
        n_agree_local = 0
        disagree = []
        for v in V:
            for B in Bs:
                cs = {lv: C[(lv, v, B)]["mode_count_protocol_1e6"] for lv in ("ex_r2", "R1", "R2", "em_r2")}
                if len(set(cs.values())) > 1:
                    n_level_change += 1
                lc = LC[(v, B)]["mode_count_protocol_1e6"]
                if all(x == lc for x in cs.values()):
                    n_agree_local += 1
                else:
                    disagree.append({"variant": v, "B": B, "hpc": cs, "local": lc})
        # 2. contact claims
        def cnt(v, B, lv="R1"):
            return C[(lv, v, B)]["mode_count_protocol_1e6"]

        def late(v, B, lv="R1"):
            c = C[(lv, v, B)]
            return c["basins_window"]["masses"][-1], c["basins_window"]["se_batch"][-1]

        gate_vs_mean = []
        for a in ("0.4", "0.2", "0.15"):
            for fld in ("mid", "p1"):
                for B in Bs:
                    g, mn = f"gcontact_a{a}_f{fld}", f"gmean_a{a}_f{fld}"
                    lg, lgs = late(g, B)
                    lm, lms = late(mn, B)
                    gate_vs_mean.append({"a": float(a), "field": fld, "B": B,
                                         "count_gated": cnt(g, B), "count_mean_contact": cnt(mn, B),
                                         "late_gated": lg, "late_gated_se": lgs,
                                         "late_mean_contact": lm, "late_mean_contact_se": lms,
                                         "local_late_gated": LC[(g, B)]["basin_masses"][-1],
                                         "local_late_mean_contact": LC[(mn, B)]["basin_masses"][-1],
                                         "local_count_gated": LC[(g, B)]["mode_count_protocol_1e6"],
                                         "local_count_mean_contact": LC[(mn, B)]["mode_count_protocol_1e6"]})
        a04_changes = [r for r in gate_vs_mean if r["a"] == 0.4 and r["count_gated"] != r["count_mean_contact"]]
        protected = [r for r in gate_vs_mean if r["count_gated"] > r["count_mean_contact"]]
        # contact contributions (R1) and their dt sensitivity
        contr = {}
        for c in A["contrasts"]:
            if not c["kind"].startswith("contact_contribution"):
                continue
            key = (c["minus"][0], float(c["B"]))
            contr.setdefault(key, {})[c["level"]] = c
        cc_rows = []
        for (g, B), lv in sorted(contr.items()):
            r1 = lv["R1"]
            row = {"variant": g, "B": B, "diff_R1": r1["basin_mass_diff"], "se_R1": r1["se_batch"],
                   "max_abs_diff_R1": max(abs(x) for x in r1["basin_mass_diff"])}
            for other in ("ex_r2", "em_r2"):
                if other in lv:
                    row[f"max_abs_{other}_minus_R1"] = max(abs(x - y) for x, y in zip(lv[other]["basin_mass_diff"], r1["basin_mass_diff"]))
            cc_rows.append(row)
        # dt effects on window basins
        dt_ex = max(max(abs(x) for x in e["window"]["ex_r2_minus_R1"]["diff"]) for e in A["dt_effects"])
        dt_em = max(max(abs(x) for x in e["window"]["em_r2_minus_ex_r2"]["diff"]) for e in A["dt_effects"])
        dt_R2R1_z = max(e["window"]["R2_minus_R1"]["max_abs_z"] for e in A["dt_effects"])
        dt_by_B = {}
        for e in A["dt_effects"]:
            B = float(e["B"])
            x = max(max(abs(v) for v in e["window"]["ex_r2_minus_R1"]["diff"]),
                    max(abs(v) for v in e["window"]["em_r2_minus_ex_r2"]["diff"]),
                    max(abs(v) for v in e["window"]["em_r2_minus_R1"]["diff"]))
            dt_by_B[B] = max(dt_by_B.get(B, 0.0), x)
        # contact contribution at B = 1, a = 0.4, mid, vs local
        c1 = contr[("gcontact_a0.4_fmid", 1.0)]["R1"]
        lc1 = next(x for x in la["contrasts"] if x["kind"].startswith("contact_contribution")
                   and x["minus"][0] == "gcontact_a0.4_fmid" and float(x["B"]) == 1.0)
        # mean-field overestimate of the last basin at B = 1, a = 0.4 (gated law), and share removed by the indicator
        cg = C[("R1", "gcontact_a0.4_fmid", 1.0)]
        mf_over = cg["basins_window"]["mean_field"][-1] - cg["basins_window"]["masses"][-1]
        share = -c1["basin_mass_diff"][-1] / mf_over
        # frozen-gate vs product law, 12 cells per anchor (a x B <= 20), window basins
        fz = []
        for a in ("0.4", "0.2", "0.15"):
            for B in Bs:
                c = C[("R1", f"gcontact_a{a}_fmid", B)]
                Mw = c["basins_window"]["masses"]
                prod = sum(abs(x - y) for x, y in zip(Mw, c["product_limit_masses"]))
                frz = sum(abs(x - y) for x, y in zip(Mw, c["frozen_surrogate_masses"]))
                lcell = LC[(f"gcontact_a{a}_fmid", B)]
                fz.append({"a": float(a), "B": B, "abs_err_product_window": prod, "abs_err_frozen_window": frz,
                           "abs_err_product_extended": c["abs_err_product_sum_extended"],
                           "abs_err_frozen_extended": c["abs_err_frozen_surrogate_sum_extended"],
                           "local_abs_err_product": lcell["abs_err_product_sum"],
                           "local_abs_err_frozen": lcell["abs_err_frozen_surrogate_sum"],
                           "frozen_better_window": frz < prod,
                           "frozen_better_extended": c["abs_err_frozen_surrogate_sum_extended"] < c["abs_err_product_sum_extended"]})
        # field location
        fld_change = []
        fld_mass = 0.0
        for gv in ("gcontact_a0.4", "gcontact_a0.2", "gcontact_a0.15", "gmean_a0.4", "gmean_a0.2", "gmean_a0.15", "gnone_a0.4"):
            for B in Bs:
                cm, cp = C[("R1", gv + "_fmid", B)], C[("R1", gv + "_fp1", B)]
                if cm["mode_count_protocol_1e6"] != cp["mode_count_protocol_1e6"]:
                    fld_change.append({"variant": gv, "B": B, "mid": cm["mode_count_protocol_1e6"], "p1": cp["mode_count_protocol_1e6"]})
                if gv.startswith("gcontact"):
                    fld_mass = max(fld_mass, max(abs(x - y) for x, y in zip(cp["basins_window"]["masses"], cm["basins_window"]["masses"])))
        s1 = A["s1_widths_B1"]["rows"]["R1"]
        s1_rows = [{"t_G1": r["t_G1"], "ratio": r["ratio"], "ratio_se": r["ratio_se"], "shift": r["vertex_shift"],
                    "s1_ratio": r["s1_ratio"], "s1_ratio_se": r["s1_ratio_se"], "s1_shift": r["s1_vertex_shift"],
                    "z_ratio": (r["ratio"] - r["s1_ratio"]) / math.hypot(r["ratio_se"], r["s1_ratio_se"])} for r in s1]
        link = A["local_n3_link"]
        res["anchors"][an] = {
            "m": m, "N_paths": A["N_paths"], "budgets": Bs, "n_variants": len(V),
            "mode_counts": {"n_cells": n_cells, "n_cells_count_changes_across_4_levels(ex_r2,R1,R2,em_r2)": n_level_change,
                            "n_cells_all_levels_equal_local": n_agree_local, "disagreements": disagree},
            "gate_vs_mean_contact": gate_vs_mean,
            "a0.4_count_changes_gated_vs_mean": a04_changes,
            "cells_gate_keeps_extra_mode": protected,
            "contact_contribution_rows": cc_rows,
            "contact_contribution_a0.4_mid_B1": {"hpc_R1": c1["basin_mass_diff"], "hpc_se": c1["se_batch"],
                                                 "hpc_em_r2": contr[("gcontact_a0.4_fmid", 1.0)]["em_r2"]["basin_mass_diff"],
                                                 "local": lc1["basin_mass_diff"], "local_se": lc1["se_batch"]},
            "meanfield_overestimate_last_basin_B1_a0.4": mf_over,
            "share_removed_by_indicator_B1_a0.4": share,
            "dt_effects_window": {"max_abs_ex_r2_minus_R1": dt_ex, "max_abs_em_r2_minus_ex_r2": dt_em,
                                  "max_abs_z_R2_minus_R1": dt_R2R1_z, "max_abs_any_by_B": dt_by_B},
            "contact_contribution_max_abs_by_a_B": {f"{r['variant']}|B{r['B']:g}": r["max_abs_diff_R1"] for r in cc_rows},
            "frozen_vs_product": {"rows": fz, "n": len(fz),
                                  "n_frozen_better_window": sum(r["frozen_better_window"] for r in fz),
                                  "n_frozen_better_extended": sum(r["frozen_better_extended"] for r in fz)},
            "field_location": {"n_cells": 7 * len(Bs), "count_changes_p1_vs_mid": fld_change,
                               "max_abs_gated_basin_change_p1_minus_mid": fld_mass},
            "s1_widths_B1_R1": s1_rows,
            "local_link": {"max_abs_z": link["max_abs_z"], "n_abs_z_gt_3": link["n_abs_z_gt_3"], "n_basins": link["n_basins"]},
        }
    O_N3.write_text(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig_n5(A5: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import exact_m_prr_upgrade_core as core
    core.apply_prr_style()
    rows = A5["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), layout="constrained")
    sty = {"em": dict(color=core.OI_VERMILLION, marker="o", mfc=core.OI_VERMILLION, label="Euler–Maruyama (production)"),
           "ex": dict(color=core.OI_BLUE, marker="s", mfc="white", label="exact OU transitions"),
           "exl": dict(color=core.OI_GREEN, marker="^", mfc=core.OI_GREEN,
                       label=r"exact OU, relabelled by $+\Delta t/2$")}
    keys = {"em": "em_r2_minus_limit", "ex": "ex_r2_minus_limit", "exl": "ex_r2_relabelled_minus_limit"}
    for ax, fams, title in ((axes[0, 0], ("basin_masses", "late_rel_prominence_smoothed"), "(a) basin masses and last-mode prominence"),
                            (axes[0, 1], ("peak_times_smoothed",), "(b) peak times")):
        rr = [r for r in rows if r["family"] in fams]
        rr.sort(key=lambda r: (fams.index(r["family"]), ORDER5.index(r["budget"])))
        x = np.arange(len(rr))
        for s, off in (("em", -0.22), ("ex", 0.0), ("exl", 0.22)):
            y = [r[keys[s]] / r["ci95_halfwidth_dt1e-3"] for r in rr]
            ax.plot(x + off, y, ls="none", ms=3.2, mew=0.7, **{k: v for k, v in sty[s].items() if k != "label"})
        ax.axhspan(-1, 1, color="0.88", lw=0, zorder=0)
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        yt = [-30, -10, -3, -1, 0, 1, 3, 10, 30]
        ax.set_yticks(yt)
        ax.set_yticklabels([str(v).replace("-", "\u2212") for v in yt])
        ax.set_ylim(-45, 45)
        ax.set_ylabel(r"bias at $\Delta t=10^{-3}$ / 95% half-width")
        # group separators and labels
        prev = None
        starts = []
        for i, r in enumerate(rr):
            g = (r["family"], r["budget"])
            if g != prev:
                starts.append((i, g))
                if i:
                    ax.axvline(i - 0.5, color="0.75", lw=0.5)
                prev = g
        ticks, labs = [], []
        for k, (i, g) in enumerate(starts):
            j = starts[k + 1][0] if k + 1 < len(starts) else len(rr)
            ticks.append(0.5 * (i + j - 1))
            if g[0] == "late_rel_prominence_smoothed":
                labs.append("")
            else:
                labs.append(LABEL5_2L[g[1]])
        if "late_rel_prominence_smoothed" in fams:
            ip = [i for i, r in enumerate(rr) if r["family"] == "late_rel_prominence_smoothed"]
            ax.axvline(ip[0] - 0.5, color="0.3", lw=0.8)
            ax.text(0.5 * (ip[0] + ip[-1]), -0.13, "prominence\n(same 4 cells)", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, fontsize=6.0)
        ax.set_xlim(-0.7, len(rr) - 0.3)
        ax.set_title(title, loc="left")
    # (c) peak-time advance vs t_peak
    ax = axes[1, 0]
    dec = A5["peak_time_decomposition"]["rows"]
    t = np.array([r["t_peak"] for r in dec])
    ax.plot(t, 1e3 * np.array([r["advance_em"] for r in dec]), ls="none", ms=3.4, **{k: v for k, v in sty["em"].items() if k != "label"})
    ax.plot(t, 1e3 * np.array([r["advance_exactOU"] for r in dec]), ls="none", ms=3.4, mew=0.8,
            **{k: v for k, v in sty["ex"].items() if k != "label"})
    tt = np.linspace(0.6, 3.1, 50)
    ax.plot(tt, 0.5 * (1 + tt), color=core.OI_VERMILLION, lw=0.8, ls=":")
    ax.plot(tt, 0.5 + 0 * tt, color=core.OI_BLUE, lw=0.8, ls=":")
    ax.text(0.7, 1.35, r"EM: $\frac{\Delta t}{2}(1+\gamma t)$", color=core.OI_VERMILLION, ha="left", va="bottom", fontsize=7)
    ax.text(0.7, 0.58, r"exact OU: $\frac{\Delta t}{2}$", color=core.OI_BLUE, ha="left", va="bottom", fontsize=7)
    ax.set_xlabel(r"peak time $t_{\rm peak}$ ($1/\gamma$)")
    ax.set_ylabel(r"peak advance at $\Delta t=10^{-3}$ ($10^{-3}$)")
    ax.set_ylim(0, 2.3)
    ax.set_title("(c) peak-time offset: end-point exposure + EM clock", loc="left")
    # (d) boundary cell
    ax = axes[1, 1]
    R = A5["boundary_cell"]["resampling"]
    order = [("em_r2", r"EM $10^{-3}$"), ("em_r1", r"EM $5{\times}10^{-4}$"), ("ex_r1", r"exact $5{\times}10^{-4}$"),
             ("ex_b4", r"exact $1.25{\times}10^{-4}$")]
    for k, (s, lab) in enumerate(order):
        for n, dx, col in ((500000, -0.12, core.OI_ORANGE), (1000000, 0.12, core.OI_PURPLE)):
            r = R[f"{s}_N{n}"]
            lo, hi = r["wilson95"]
            ax.errorbar(k + dx, r["p"], yerr=[[r["p"] - lo], [hi - r["p"]]], color=col, marker="o", ms=3, capsize=1.5, lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o[1] for o in order], fontsize=6.2)
    ax.set_ylabel(r"$P(\text{protocol count}=3)$")
    ax.set_ylim(0.975, 1.0005)
    ax.set_title(r"(d) boundary cell $(3,0.175,0.5)$", loc="left")
    ax.legend(handles=[Line2D([], [], color=core.OI_ORANGE, marker="o", ms=3, lw=0.8, label=r"$5\times10^5$ walkers"),
                       Line2D([], [], color=core.OI_PURPLE, marker="o", ms=3, lw=0.8, label=r"$10^6$ walkers")],
              loc="lower right", frameon=False, fontsize=6.5)
    handles = [Line2D([], [], ls="none", ms=3.5, **sty[s]) for s in ("em", "ex", "exl")]
    handles.append(matplotlib.patches.Patch(color="0.88", label="within the 95% sampling half-width"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.3, frameon=False)
    out = core.save_figure(fig, core.FIGURES / "fb_hpc_n5full")
    plt.close(fig)
    return out


def fig_n3(A3: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import exact_m_prr_upgrade_core as core
    core.apply_prr_style()
    col = {0.4: core.OI_BLUE, 0.2: core.OI_ORANGE, 0.15: core.OI_GREEN}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    for ax, an, title in ((axes[0, 0], "m2_eps0.1", r"(a) late basin, $(m,\varepsilon)=(2,0.1)$"),
                          (axes[0, 1], "m3_eps0.1", r"(b) last basin, $(m,\varepsilon)=(3,0.1)$")):
        G = [r for r in A3["anchors"][an]["gate_vs_mean_contact"] if r["field"] == "mid"]
        for a in (0.4, 0.2, 0.15):
            rr = sorted([r for r in G if r["a"] == a], key=lambda r: r["B"])
            B = np.array([r["B"] for r in rr])
            ax.plot(B, [r["late_gated"] for r in rr], color=col[a], marker="o", ms=3.2, lw=1.0)
            ax.plot(B, [r["late_mean_contact"] for r in rr], color=col[a], marker="o", mfc="white", ms=3.2, lw=0.9, ls="--")
            ax.plot(B, [r["local_late_gated"] for r in rr], color="k", marker="x", ms=3.0, ls="none", mew=0.6)
            ax.plot(B, [r["local_late_mean_contact"] for r in rr], color="k", marker="x", ms=3.0, ls="none", mew=0.6)
            for r in rr:
                if r["count_gated"] > r["count_mean_contact"]:
                    ax.annotate(f"{r['count_gated']} vs {r['count_mean_contact']}", (r["B"], r["late_gated"]),
                                textcoords="offset points", xytext=(-5, 5 if a == 0.15 else -10), fontsize=6, color=col[a], ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 8, 20])
        ax.set_xticklabels(["1", "4", "8", "20"])
        ax.set_xlabel(r"budget $B$")
        ax.set_ylabel("last window basin mass")
        ax.set_title(title, loc="left")
    # (c) contact contribution vs dt effects
    ax = axes[1, 0]
    for k, (an, mk) in enumerate((("m2_eps0.1", "o"), ("m3_eps0.1", "s"))):
        A = A3["anchors"][an]
        for a in (0.4, 0.2, 0.15):
            rr = sorted([r for r in A["contact_contribution_rows"] if r["variant"] == f"gcontact_a{a}_fmid"], key=lambda r: r["B"])
            ax.plot([r["B"] * (1 + 0.06 * k) for r in rr], [r["max_abs_diff_R1"] for r in rr], color=col[a], marker=mk,
                    ms=3.2, lw=0.9, mfc=col[a] if k == 0 else "white")
        dB = A["dt_effects_window"]["max_abs_any_by_B"]
        Bs = sorted(float(b) for b in dB)
        ax.plot([b * (1 + 0.06 * k) for b in Bs], [dB[b] if b in dB else dB[str(b)] for b in Bs], color="0.35", marker=mk,
                ms=3.0, lw=0.8, ls=":", mfc="0.35" if k == 0 else "white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 8, 20])
    ax.set_xticklabels(["1", "4", "8", "20"])
    ax.set_xlabel(r"budget $B$")
    ax.set_ylabel(r"max. $|\Delta|$ of a window basin mass")
    ax.set_title("(c) contact contribution vs time-step effect", loc="left")
    # (d) frozen-gate vs product law
    ax = axes[1, 1]
    for k, (an, mk) in enumerate((("m2_eps0.1", "o"), ("m3_eps0.1", "s"))):
        for r in A3["anchors"][an]["frozen_vs_product"]["rows"]:
            ax.plot(r["abs_err_product_window"], r["abs_err_frozen_window"], marker=mk, color=col[r["a"]], ms=3.4,
                    mfc=col[r["a"]] if k == 0 else "white", ls="none", mew=0.8)
    lim = [3e-3, 1.0]
    ax.plot(lim, lim, color="0.5", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("summed basin error, product law")
    ax.set_ylabel("summed basin error, frozen-gate law")
    ax.set_title(r"(d) frozen-gate vs product law, $B\leq 20$", loc="left")
    handles = [Line2D([], [], color=col[a], lw=1.2, label=f"$a={a:g}$") for a in (0.4, 0.2, 0.15)]
    handles += [Line2D([], [], color="k", marker="o", ms=3, lw=1.0, label="(a,b) contact indicator"),
                Line2D([], [], color="k", marker="o", mfc="white", ms=3, lw=0.9, ls="--", label=r"(a,b) mean contact $c_a(t)$"),
                Line2D([], [], color="k", marker="o", ms=3, ls="none", label=r"(c,d) $m=2$"),
                Line2D([], [], color="k", marker="s", mfc="white", ms=3, ls="none", label=r"(c,d) $m=3$"),
                Line2D([], [], color="k", marker="x", ms=3, ls="none", label=r"local $10^6$-path run"),
                Line2D([], [], color="0.35", ls=":", marker="o", ms=3, label=r"largest $\Delta t$ effect (exact/EM/Richardson)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.2, frameon=False)
    out = core.save_figure(fig, core.FIGURES / "fb_hpc_n3full")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Snippets
# ----------------------------------------------------------------------------

def sci(x, d=2):
    if x == 0:
        return "0"
    mnt, e = f"{x:.{d - 1}e}".split("e")
    return f"{mnt}\\times10^{{{int(e)}}}"


JOBS = {"N5full": "5768127 (simulate+analyze, COMPLETED 00:06:01, 0.100 node-h) + 5768647 (deterministic re-analysis, 00:00:28, 0.008 node-h)",
        "N3full": "5768126 (simulate+analyze, COMPLETED 00:03:23, 0.056 node-h) + 5768647 (deterministic re-analysis, 00:00:28, 0.008 node-h)"}


def snip_n5(A5: dict) -> str:
    D = load(P_N5)
    c = A5["counts"]
    bm, pt, pr = c["basin_masses"], c["peak_times_smoothed"], c["late_rel_prominence_smoothed"]
    tot = c["total"]
    pd = A5["peak_time_decomposition"]
    pm = A5["prominence"]["m5_anchor"]
    pa = A5["prominence"]["m3_anchor"]
    bc = A5["boundary_cell"]
    R = bc["resampling"]
    lz = max(A5["local_N5_max_abs_z_em_r2"].values())
    ordr = [(k, r["em_order_ratio"]) for r in A5["rows"] for k in [r["family"]] if r["em_order_ratio"] is not None]
    mo = [x for k, x in ordr if k == "basin_masses"]
    Pa = rel(O_N5)
    Pj = rel(P_N5)
    L = []
    L.append("% " + "=" * 69)
    L.append("% HPC_N5full -- Isambard 3 (partition grace): time-step ladder at 3e6 paths per cell")
    L.append(f"% Acceptance (plan N5, |f(dt=1e-3) - limit| < 95% half-width), production EM: masses {bm['n_pass_em']}/14,")
    L.append(f"%   peak times {pt['n_pass_em']}/14, prominence {pr['n_pass_em']}/4 -> NOT MET as stated; exact OU at the same dt:")
    L.append(f"%   {bm['n_pass_ex']}/14, {pt['n_pass_ex']}/14, {pr['n_pass_ex']}/4; exact OU relabelled by +dt/2: {bm['n_pass_ex_relabelled']}/14, "
             f"{pt['n_pass_ex_relabelled']}/14, {pr['n_pass_ex_relabelled']}/4.")
    L.append(f"% Data:   {Pj} (fetched); {Pa} (derived by code/fb_hpc_n5n3_report.py)")
    L.append("% Driver: code/fb_hpc_n5full.py (seed base 20260923, tag 95); figure artifacts/figures/fb_hpc_n5full.pdf")
    L.append(f"% Jobs:   {JOBS['N5full']}")
    L.append("% Writer notes: the pass/fail criterion scales with N (a fixed O(dt) bias fails once 1.96 SE < bias);")
    L.append("%   quote biases in absolute units.  The dt/2 relabelling is a derived diagnostic (not run on the cluster).")
    L.append("% " + "=" * 69)
    L.append(r"\paragraph{Time-step convergence at scale.}")
    L.append(r"We repeated the time-step ladder of SM Section~\smref{sec:sm-N5} on Isambard~3 with $3\times10^6$ unkilled paths per cell "
             r"(three seeds of $10^6$), about five times the local ensemble. All schemes share one set of Gaussian increments "
             r"at $h=5\times10^{-4}$: the production Euler--Maruyama (EM) chain and the exact Ornstein--Uhlenbeck (OU) chain at "
             r"$\Delta t\in\{5\times10^{-4},10^{-3},2\times10^{-3}\}$, and the exact chain refined inside every step by exact bridge "
             r"points to $h/2$ and $h/4$. We take as reference the second-order Richardson limit of the bridge-refined exact ladder. "
             r"The EM limit agrees with it: for the basin masses the largest difference is "
             + f"${sci(bm['max_abs_em_R2_minus_exb_R2'])}$ ($|z|\\le{bm['max_abs_z_em_R2_vs_exb_R2']:.1f}$), "
             + r"and for the peak times it is "
             + f"${sci(pt['max_abs_em_R2_minus_exb_R2'])}$.")
    L.append(f"% src: {Pj}#cells.*.N_paths (3000000); #cells.*.budgets.*.functionals.*.{{em_vs_exb_richardson2_diff, em_vs_exb_richardson2_z}} "
             f"(max over families: {Pa}#counts.{{basin_masses,peak_times_smoothed}}.{{max_abs_em_R2_minus_exb_R2, max_abs_z_em_R2_vs_exb_R2}}; "
             f"peak-time |z| reaches {pt['max_abs_z_em_R2_vs_exb_R2']:.1f} on paired SEs of ~1e-6)")
    L.append(r"All schemes converge at first order (EM successive-difference ratios of the basin masses "
             + f"${min(mo):.2f}$--${max(mo):.2f}$).")
    L.append(f"% src: {Pj}#cells.*.budgets.*.functionals.basin_masses[*].em_order_ratio")
    L.append("")
    L.append(r"At $\Delta t=10^{-3}$ the production chain is biased by at most "
             + f"${sci(bm['max_abs_em'])}$ in a basin mass, ${sci(pt['max_abs_em'])}$ in a peak time "
             + r"(a tenth of a $0.02$ bin) and "
             + f"${sci(pr['max_abs_em'])}$ in a last-mode relative prominence. "
             + r"These biases are deterministic and resolved (paired standard errors at most $"
             + sci(bm["max_paired_se_prod_bias_vs_exb_R2"]["value"]) + r"$, $" + sci(pt["max_paired_se_prod_bias_vs_exb_R2"]["value"])
             + r"$ and $" + sci(pr["max_paired_se_prod_bias_vs_exb_R2"]["value"]) + r"$, respectively). "
             + r"With $3\times10^6$ paths the $95\%$ sampling half-width of a single value is smaller than many of them, so "
             + r"the acceptance test of SM Section~\smref{sec:sm-N5} (bias below the half-width) passes for "
             + f"{bm['n_pass_em']} of 14 basin masses, {pt['n_pass_em']} of 14 peak times and {pr['n_pass_em']} of 4 prominences.")
    L.append(f"% src: {Pa}#counts.*.{{max_abs_em, n_pass_em}}; rows[*].{{em_r2_minus_limit, ci95_halfwidth_dt1e-3}}; "
             f"paired SE: {Pj}#cells.*.budgets.*.functionals.*.prod_bias_vs_exb_R2_se (max per family: {Pa}#counts.*.max_paired_se_prod_bias_vs_exb_R2)")
    L.append(r"The bias has two sources, and the ladder separates them. "
             r"(i)~Every scheme charges the exposure of a step to the field at the end of the step, so its law runs ahead of "
             r"the continuum law by $\Delta t/2$. For the exact chain this offset is almost the whole bias: its peak times are early by "
             + f"${sci(pd['exactOU_advance_range'][0], 3)}$--${sci(pd['exactOU_advance_range'][1], 3)}$, "
             + r"$\Delta t/2$ to within "
             + f"${sci(pd['max_abs_exactOU_advance_minus_dt2'])}$. "
             + r"After relabelling its times by $+\Delta t/2$ (peak time $+\Delta t/2$, basin mass $-\tfrac{\Delta t}{2}[f(b)-f(a)]$) "
             + r"the exact chain is within the half-width for every functional ("
             + f"{bm['n_pass_ex_relabelled']}/14, {pt['n_pass_ex_relabelled']}/14, {pr['n_pass_ex_relabelled']}/4): "
             + f"the masses to ${bm['max_abs_over_hw_ex_relabelled']:.2f}$ and the peak times to ${pt['max_abs_over_hw_ex_relabelled']:.2f}$ half-widths; "
             + r"the only visible remainder is the $m=5$ prominence, "
             + f"${sci(A5['prominence']['m5_anchor']['ex_r2_minus_limit'])}$ (${pr['max_abs_over_hw_ex']:.2f}$ half-widths), which a time shift does not change. "
             + r"(ii)~The EM chain contracts the OU coordinates by $1-\gamma\Delta t$ per step, i.e.\ at the rate "
             + r"$\gamma(1+\gamma\Delta t/2)$, so its mean path reaches a stripe earlier by a further $\gamma t\,\Delta t/2$. "
             + r"The measured EM-minus-exact advance follows $\gamma t\,\Delta t/2$ to within "
             + f"${sci(pd['max_abs_em_specific_minus_gamma_t_dt2'])}$; "
             + f"it is {100 * pd['em_specific_share_range'][0]:.0f}--{100 * pd['em_specific_share_range'][1]:.0f}\\% of the EM peak shift.")
    L.append(f"% src: {Pj}#peak_time_bias_model.rows[*].{{D_em_vs_exb_R2, D_ex_r2_vs_exb_R2, t_peak_exb_R2}}; "
             f"{Pa}#peak_time_decomposition.{{exactOU_advance_range, max_abs_exactOU_advance_minus_dt2, max_abs_em_specific_minus_gamma_t_dt2, em_specific_share_range}}; "
             f"{Pa}#counts.*.{{n_pass_ex_relabelled, max_abs_over_hw_ex_relabelled}} (relabelling: mass - (dt/2)[f(b)-f(a)] from "
             f"{Pj}#cells.*.budgets.*.bin_mass.ex_r2 and cuts_idx; peak + dt/2; prominence unchanged)")
    L.append(r"This answers what the local failures were. The prominence failures are Euler--Maruyama discretisation and "
             r"exact OU transitions remove them. The $m=5$ last-mode prominence is "
             + f"${pm['limit']:.4f}$; " + r"the EM chain overstates it by "
             + f"${sci(pm['em_r2_minus_limit'])}$ " + r"(the local run found $" + sci(-pm['local_plan_D_em']) + r"$), "
             + r"the exact chain at the same step by only "
             + f"${sci(pm['ex_r2_minus_limit'])}$, "
             + f"below the half-width ${sci(pm['hw'])}$. "
             + r"The same holds for the $m=3$ anchor ($" + sci(pa['em_r2_minus_limit']) + r"$ against $" + sci(pa['ex_r2_minus_limit'], 2)
             + r"$, half-width $" + sci(pa['hw']) + r"$). "
             + r"The peak times, early by "
             + f"${sci(pd['em_advance_range'][0])}$--${sci(pd['em_advance_range'][1])}$ "
             + r"in the EM chain, are only partly an EM effect: exact OU transitions remove the $\gamma t\,\Delta t/2$ part and leave the "
             + r"$\Delta t/2$ offset of the end-point exposure rule, which any scheme with that rule shares at the same step and which a "
             + r"relabelling (or a midpoint exposure rule) removes. Neither effect changes a protocol mode count: at every scheme and "
             + r"both histogram sizes the count is $m$, except at $(3,0.1,3.57)$, where it is $2$ at every scheme "
             + r"(prominence $" + f"{A5['prominence']['m3_threshold']['limit']:.4f}" + r"$: the cell sits at the visibility threshold $B_{5\%}$ and the last mode is just below the floor).")
    L.append(f"% src: {Pa}#prominence.{{m5_anchor,m3_anchor}}.{{limit, em_r2_minus_limit, ex_r2_minus_limit, hw, local_plan_D_em}}; "
             f"local: {rel(P_N5L)}#cells.m5_z08_eps0.1.budgets.m5_anchor.functionals.late_rel_prominence_smoothed.prod_bias_D (-1.71e-3, "
             f"half-width 1.21e-3); HPC plan_D {pm['plan_D_em']:.3e} +- {pm['plan_D_se']:.1e} vs local {pm['local_plan_D_em']:.3e} +- "
             f"{pm['local_plan_D_se']:.1e} (z = {pm['z_hpc_vs_local_plan_D']:.2f}); mode counts: "
             f"{Pj}#cells.*.budgets.*.functionals.protocol_mode_count_N{{500000,1000000}}.values; {Pa}#protocol_mode_count_by_scheme")
    sp = bc["stored_pair"]
    L.append(r"In the boundary cell $(3,0.175,0.5)$ the stored pair of $5\times10^5$-walker runs gave three modes at $\Delta t$ "
             r"and two at $\Delta t/2$. The exact law puts the last-mode prominence at "
             + f"${bc['late_rel_prominence_limit']:.4f}\\pm{bc['late_rel_prominence_hw']:.4f}$, "
             + r"just above the $5\%$ floor, and the production step changes it by $" + sci(abs(pr_row_em(A5, 'm3_phase_boundary'))) + r"$. "
             + r"Among multinomial $5\times10^5$-walker histograms drawn from the exact laws ($2000$ replicas each) the protocol "
             + r"returns three modes with probability "
             + f"${R['em_r2_N500000']['p']:.3f}$ " + r"(EM, $10^{-3}$), "
             + f"${R['em_r1_N500000']['p']:.4f}$ " + r"(EM, $5\times10^{-4}$) and "
             + f"${R['ex_b4_N500000']['p']:.3f}$ " + r"(finest exact law). "
             + r"Halving the step does not change this probability; the observed three-then-two pattern has probability "
             + f"${bc['p_observed_pattern_3_then_2_at_5e5']:.3f}$" + r" and is a sampling event in a floor-crossing cell, "
             + r"not a time-step effect (stored prominences $" + f"{sp['dt']['late_relative_prominence']:.3f}" + r"$ and $"
             + f"{sp['dt_half']['late_relative_prominence']:.3f}" + r"$ straddle the exact value).")
    L.append(f"% src: {Pj}#boundary_flip.{{stored_pair, resampling.{{em_r2_N500000, em_r1_N500000, ex_b4_N500000}}.{{p_mode_count_eq_m, wilson95}}, "
             f"p_observed_pattern_3_then_2_at_5e5}}; {Pj}#cells.m3_eps0.175.budgets.m3_phase_boundary.functionals.late_rel_prominence_smoothed."
             f"{{exb_richardson2, prod_value_ci95_halfwidth, values.em_r2}}; Wilson 95%: em_r2 [0.9858, 0.9943], em_r1 [0.9828, 0.9923]")
    L.append(r"The production-chain values agree with the independent local run (largest $|z|=" + f"{lz:.1f}" + r"$).")
    L.append(f"% src: {Pj}#local_N5_comparison.*.max_abs_z_em_r2")
    L.append("")
    L.append("% Figure caption (fb_hpc_n5full): Time-step bias on 3e6 common paths per cell (Isambard 3). (a,b) Bias at dt = 1e-3 of the")
    L.append("% basin masses and last-mode relative prominences (a) and of the smoothed peak times (b), in units of the 95% sampling")
    L.append("% half-width of a single dt = 1e-3 value (grey band = acceptance), for the production Euler-Maruyama chain (filled circles),")
    L.append("% the exact OU chain at the same step (open squares) and the exact chain with times relabelled by +dt/2 (triangles).")
    L.append("% Reference: second-order Richardson limit of the bridge-refined exact ladder. (c) Peak advance (limit minus dt = 1e-3 value)")
    L.append("% against peak time; dotted: (dt/2)(1 + gamma t) and dt/2. (d) Boundary cell (3,0.175,0.5): probability that the protocol")
    L.append("% counts three modes in multinomial histograms of 5e5 and 1e6 walkers drawn from the laws at four steps (2000 replicas, Wilson 95%).")
    return "\n".join(L) + "\n"


def pr_row_em(A5, lab):
    r = next(x for x in A5["rows"] if x["budget"] == lab and x["family"] == "late_rel_prominence_smoothed")
    return r["em_r2_minus_limit"]


def snip_n3(A3: dict) -> str:
    Pj = rel(P_N3)
    Pa = rel(O_N3)
    Pl = rel(P_N3L)
    a2, a3 = A3["anchors"]["m2_eps0.1"], A3["anchors"]["m3_eps0.1"]

    def gv(A, a, B, fld="mid"):
        return next(r for r in A["gate_vs_mean_contact"] if r["a"] == a and r["B"] == B and r["field"] == fld)

    def cc(A, a, B):
        return next(r for r in A["contact_contribution_rows"] if r["variant"] == f"gcontact_a{a}_fmid" and r["B"] == B)

    L = []
    L.append("% " + "=" * 69)
    L.append("% HPC_N3full -- Isambard 3 (partition grace): contact / field factorial, GPT-6 full specification")
    L.append("% exact OU at dt = 5e-4, 1e-3, 2e-3 (+ Richardson R1, R2) and production EM at 1e-3; 3 seeds x 1e6 per anchor; B <= 20")
    L.append("% NOT covered here (still local only): B = 1e2..1e4, the boundary-tangent (frozen-gate) arm, the r_par0 = 0.35 arm.")
    L.append(f"% Data:   {Pj} (fetched); {Pa} (derived by code/fb_hpc_n5n3_report.py); local reference {Pl}")
    L.append("% Driver: code/fb_hpc_n3full.py (seed base 20260923, tag 96); figure artifacts/figures/fb_hpc_n3full.pdf")
    L.append(f"% Jobs:   {JOBS['N3full']}")
    L.append("% " + "=" * 69)
    L.append(r"\paragraph{Contact factorial with exact transitions.}")
    L.append(r"We repeated the contact and field-location factorial of SM Section~\smref{sec:sm-N3} on Isambard~3 for $B\le20$ with "
             r"$3\times10^6$ common paths per anchor ($(m,\varepsilon)=(2,0.1)$, $(3,0.1)$; three seeds of $10^6$). Each of the "
             r"$14$ kernels was evaluated with exact OU transitions at $\Delta t=5\times10^{-4}$, $10^{-3}$, $2\times10^{-3}$, "
             r"their Richardson limits, and the production EM chain at $10^{-3}$.")
    L.append(f"% src: {Pj}#anchors.*.{{N_paths, variants, budgets}}; #levels")
    L.append(r"The time step is irrelevant for the contact conclusions. In all $" + f"{a2['mode_counts']['n_cells'] + a3['mode_counts']['n_cells']}"
             + r"$ (kernel, $B$) cells the protocol count is the same for the exact $\Delta t=10^{-3}$ chain, both Richardson limits and the EM chain, and it equals the count of the local "
             + r"$10^6$-path EM run in " + f"{a2['mode_counts']['n_cells_all_levels_equal_local'] + a3['mode_counts']['n_cells_all_levels_equal_local']}"
             + r" of them. The step moves a window basin mass by at most $"
             + sci(max(a2['dt_effects_window']['max_abs_ex_r2_minus_R1'], a3['dt_effects_window']['max_abs_ex_r2_minus_R1']))
             + r"$ in the exact chain and by $"
             + sci(max(a2['dt_effects_window']['max_abs_em_r2_minus_ex_r2'], a3['dt_effects_window']['max_abs_em_r2_minus_ex_r2']))
             + r"$ between EM and exact transitions, against contact contributions (largest basin change for a given $a$ and $B$) of $"
             + f"{min(min(A['contact_contribution_max_abs_by_a_B'].values()) for A in (a2, a3)):.3f}$--$"
             + f"{max(max(A['contact_contribution_max_abs_by_a_B'].values()) for A in (a2, a3)):.2f}$ "
             + r"(figure fb\_hpc\_n3full, panel c). The production-chain basin masses agree with the local run to $|z|\le"
             + f"{max(a2['local_link']['max_abs_z'], a3['local_link']['max_abs_z']):.1f}" + r"$ ("
             + f"{a2['local_link']['n_abs_z_gt_3'] + a3['local_link']['n_abs_z_gt_3']}" + r" of "
             + f"{a2['local_link']['n_basins'] + a3['local_link']['n_basins']}" + r" basins above $3$).")
    L.append(f"% src: {Pj}#anchors.*.mode_count_table; {Pa}#anchors.*.mode_counts; {Pa}#anchors.*.dt_effects_window "
             f"(from {Pj}#anchors.*.dt_effects[*].window.{{ex_r2_minus_R1, em_r2_minus_ex_r2}}.diff; resolved: paired SE ~1e-7); "
             f"{Pj}#anchors.*.local_n3_link.{{max_abs_z, n_abs_z_gt_3, n_basins}}; contact contributions: {Pa}#anchors.*.contact_contribution_max_abs_by_a_B "
             f"(from {Pj}#anchors.*.contrasts[level=R1, kind=contact_contribution*].basin_mass_diff)")
    c1 = a2["contact_contribution_a0.4_mid_B1"]
    c13 = a3["contact_contribution_a0.4_mid_B1"]
    L.append(r"The contact statements of Section~\ref{sec:num-contact} for $B\le20$ are reproduced. "
             r"At the production radius $a=0.4$ the contact indicator never changes the protocol count relative to its mean $c_a(t)$ "
             r"(" + f"{len(a2['a0.4_count_changes_gated_vs_mean']) + len(a3['a0.4_count_changes_gated_vs_mean'])}" + r" changes in $16$ cells, "
             r"both field locations); at $B=1$ it changes the window basin masses by $("
             + ",".join(f"{x:.4f}" for x in c1["hpc_R1"]) + r")$ at $m=2$ and $("
             + ",".join(f"{x:.4f}" for x in c13["hpc_R1"]) + r")$ at $m=3$ (local: $("
             + ",".join(f"{x:.4f}" for x in c1["local"]) + r")$ and $("
             + ",".join(f"{x:.4f}" for x in c13["local"]) + r")$). "
             + r"This is most of the mean-field overestimate of the last basin, $"
             + f"{a2['meanfield_overestimate_last_basin_B1_a0.4']:.3f}$ ($m=2$) and ${a3['meanfield_overestimate_last_basin_B1_a0.4']:.3f}$ ($m=3$), "
             + f"of which {100 * a2['share_removed_by_indicator_B1_a0.4']:.0f}\\% and {100 * a3['share_removed_by_indicator_B1_a0.4']:.0f}\\% "
             + r"is removed by the indicator.")
    L.append(f"% src: {Pa}#anchors.*.a0.4_count_changes_gated_vs_mean (empty); {Pj}#anchors.*.contrasts[level=R1, kind=contact_contribution, "
             f"minus=[gcontact_a0.4_fmid, gmean_a0.4_fmid], B=1].basin_mass_diff; local {Pl}#anchors.m{{2,3}}.contrasts[B=1, same kind].basin_mass_diff; "
             f"{Pa}#anchors.*.{{meanfield_overestimate_last_basin_B1_a0.4, share_removed_by_indicator_B1_a0.4}} "
             f"(from {Pj}#anchors.*.cells[level=R1, variant=gcontact_a0.4_fmid, B=1].basins_window.{{mean_field, masses}})")
    g2, g15 = gv(a2, 0.2, 20.0), gv(a2, 0.15, 20.0)
    p3 = a3["cells_gate_keeps_extra_mode"]
    assert all(r["field"] == "p1" for r in p3), "m=3 midpoint protection at B<=20: rewrite the text"
    p3s = ", ".join(f"$(a,B)=({r['a']:g},{r['B']:g})$: ${r['count_gated']}$ vs ${r['count_mean_contact']}$" for r in p3)
    L.append(r"Below the production radius contact protects the late mode. At $(2,0.1,20)$ the gated late basin holds $"
             + f"{g2['late_gated']:.4f}$ ($a=0.2$) and ${g15['late_gated']:.4f}$ ($a=0.15$), against ${g2['late_mean_contact']:.4f}$ and "
             + f"${g15['late_mean_contact']:.4f}$ " + r"with $c_a(t)$ (local: $" + f"{g2['local_late_gated']:.4f}$, ${g15['local_late_gated']:.4f}$, "
             + f"${g2['local_late_mean_contact']:.4f}$, ${g15['local_late_mean_contact']:.4f}$), "
             + r"and the gated law keeps two protocol modes where the $c_a(t)$ law keeps one (both field locations). "
             + r"At $m=2$ this happens only at $B=20$ for $B\le20$ (" + f"{len(a2['cells_gate_keeps_extra_mode'])}" + r" cells, "
             + r"$a\in\{0.2,0.15\}$). At $m=3$ with the midpoint field the gated and $c_a(t)$ counts agree for every $B\le20$, "
             + r"consistent with protection only from $B=10^2$ on; with the particle-1 field they differ in "
             + f"{len(p3)}" + r" cell, " + p3s + r", where the $c_a(t)$ law loses the third mode (the field-location cell below).")
    L.append(f"% src: {Pa}#anchors.m2_eps0.1.gate_vs_mean_contact[a in (0.2,0.15), B=20, field=mid].{{late_gated, late_mean_contact, "
             f"local_late_gated, local_late_mean_contact, count_gated, count_mean_contact}}; {Pa}#anchors.*.cells_gate_keeps_extra_mode "
             f"(m=2: {[(r['a'], r['field'], r['B']) for r in a2['cells_gate_keeps_extra_mode']]}; m=3: {[(r['a'], r['field'], r['B']) for r in a3['cells_gate_keeps_extra_mode']]})")
    fz2, fz3 = a2["frozen_vs_product"], a3["frozen_vs_product"]
    r21 = next(r for r in fz2["rows"] if r["a"] == 0.4 and r["B"] == 1.0)
    r31 = next(r for r in fz3["rows"] if r["a"] == 0.4 and r["B"] == 1.0)
    L.append(r"The frozen-gate law with the measured contact states at the $t_j$ beats the product law in "
             + f"{fz2['n_frozen_better_window'] + fz3['n_frozen_better_window']}" + r" of " + f"{fz2['n'] + fz3['n']}"
             + r" (anchor, $a$, $B$) cells (window basins; for example $" + f"{r21['abs_err_product_window']:.3f}\\to{r21['abs_err_frozen_window']:.3f}$ "
             + r"at $(2,0.1,1)$ and $" + f"{r31['abs_err_product_window']:.3f}\\to{r31['abs_err_frozen_window']:.3f}$ at $(3,0.1,1)$, $a=0.4$; "
             + r"with basins over the whole time axis " + f"{fz2['n_frozen_better_extended'] + fz3['n_frozen_better_extended']}" + r" of "
             + f"{fz2['n'] + fz3['n']}" + r").")
    L.append(f"% src: {Pa}#anchors.*.frozen_vs_product.{{n_frozen_better_window, n_frozen_better_extended, rows[a=0.4,B=1]}} "
             f"(from {Pj}#anchors.*.cells[level=R1, gate=contact, field=mid].{{basins_window.masses, product_limit_masses, frozen_surrogate_masses, "
             f"abs_err_product_sum_extended, abs_err_frozen_surrogate_sum_extended}}); local: {Pl}#anchors.*.cells[*].{{abs_err_product_sum, abs_err_frozen_surrogate_sum}}")
    fc = a2["field_location"]["count_changes_p1_vs_mid"] + a3["field_location"]["count_changes_p1_vs_mid"]
    fc_s = "; ".join(f"$m={3 if r in a3['field_location']['count_changes_p1_vs_mid'] else 2}$, {r['variant'].replace('gmean', 'mean contact').replace('gcontact', 'gate').replace('_a', ', $a=')}$, $B={r['B']:g}$: ${r['mid']}\\to{r['p1']}$" for r in fc)
    s12, s13 = a2["s1_widths_B1_R1"], a3["s1_widths_B1_R1"]
    zmax = max(abs(r["z_ratio"]) for r in s12 + s13)
    L.append(r"Reading the field at particle~1 changes the gated window basin masses by at most $"
             + f"{max(a2['field_location']['max_abs_gated_basin_change_p1_minus_mid'], a3['field_location']['max_abs_gated_basin_change_p1_minus_mid']):.4f}$ "
             + r"and the protocol count in " + f"{len(fc)}" + r" of " + f"{a2['field_location']['n_cells'] + a3['field_location']['n_cells']}"
             + r" cells (" + fc_s + r", the cell found locally). At $B=1$ the particle-1 peak widths relative to the midpoint field are $"
             + ", ".join(f"{r['ratio']:.3f}" for r in s12) + r"$ ($m=2$) and $" + ", ".join(f"{r['ratio']:.3f}" for r in s13)
             + r"$ ($m=3$), and the early peaks are delayed by $" + f"{min(r['shift'] for r in s12 + s13 if r['shift'] > 0):.4f}$--$"
             + f"{max(r['shift'] for r in s12 + s13):.4f}$, " + r"matching the single-particle direct-kill records "
             + r"(widths within $|z|\le" + f"{zmax:.1f}" + r"$).")
    L.append(f"% src: {Pa}#anchors.*.field_location.{{max_abs_gated_basin_change_p1_minus_mid, count_changes_p1_vs_mid}}; "
             f"{Pj}#anchors.*.s1_widths_B1.rows.R1[*].{{ratio, ratio_se, vertex_shift, s1_ratio, s1_ratio_se}}; {Pa}#anchors.*.s1_widths_B1_R1[*].z_ratio "
             f"(largest |z| at m=3, second peak: 1.116 vs S1 1.101 +- 0.007)")
    L.append(r"What this run does not test: budgets $B\ge10^2$ (the re-entry regime and the protection at $m=3$), the "
             r"boundary-tangent frozen-gate arm (where the orthant law is approached in $\varepsilon$ but missed by up to $43$ standard "
             r"errors at $\varepsilon=0.025$), and the $r_{\parallel,0}=0.35$ arm. These remain the local $10^6$/$2\times10^5$-path results.")
    L.append(f"% src: {Pj}#budgets ([1, 4, 8, 20]); local tangent arm: {Pl}#acceptance_checks.max_abs_z_fk_minus_orthant_eps0.025_B_le_8 (43.05)")
    L.append("")
    L.append("% Figure caption (fb_hpc_n3full): Contact factorial with exact OU transitions, 3e6 common paths per anchor (Isambard 3),")
    L.append("% first-order Richardson limit of the exact ladder, window basins. (a,b) Last basin mass against B for the contact")
    L.append("% indicator (filled, solid) and its mean c_a(t) (open, dashed) at a = 0.4, 0.2, 0.15; crosses: local 1e6-path EM run;")
    L.append("% labels give protocol counts where the gate keeps an extra mode. (c) Largest contact contribution (indicator minus")
    L.append("% c_a(t)) to a window basin mass (colour; circles m = 2, squares m = 3) and largest time-step effect on any basin")
    L.append("% (grey dotted: exact dt = 1e-3 vs Richardson, EM vs exact, EM vs Richardson). (d) Summed basin error of the")
    L.append("% product law and of the frozen-gate law (contact states measured at t_j) against the exact law; B <= 20.")
    return "\n".join(L) + "\n"


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["all"])[0]
    if cmd in ("all", "derive"):
        derive_n5()
        derive_n3()
        print("[derive]", O_N5, O_N3)
    A5, A3 = load(O_N5), load(O_N3)
    if cmd in ("all", "figures"):
        print("[figures]", fig_n5(A5), fig_n3(A3))
    if cmd in ("all", "snippets"):
        (SNIP / "HPC_N5full.tex").write_text(snip_n5(A5))
        (SNIP / "HPC_N3full.tex").write_text(snip_n3(A3))
        print("[snippets]", SNIP / "HPC_N5full.tex", SNIP / "HPC_N3full.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
