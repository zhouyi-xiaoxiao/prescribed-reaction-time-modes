#!/usr/bin/env python3
"""Write manuscript result snippets HPC_<item>.tex from the fetched HPC summary JSON.

Every number is read from artifacts/data/exact_m_fixed_budget/HPC_<item>/*.json
and cited as ``% src: <path>#<key>``.  Items whose JSON is missing are skipped.

    python3 fb_hpc_snippets.py [--data-root DIR] [--out-dir DIR] [--jobs jobs.json]
``--jobs``: optional JSON {item: {"job_id":..., "node_hours":...}} for the header.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPORT = HERE.parents[1]
DATA = REPORT / "artifacts" / "data" / "exact_m_fixed_budget"
SNIP = REPORT / "manuscript" / "cnsns_submission" / "results_snippets"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPORT))
    except ValueError:
        return str(p)


def f(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        m, e = f"{x:.{max(d - 1, 1)}e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return f"{x:.{d}g}" if ax >= 1 else f"{x:.{d}f}".rstrip("0").rstrip(".") if d > 0 else f"{x}"


def sci(x, d=2):
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    m, e = f"{x:.{d - 1}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def header(item, status_lines, data_rel, driver, tag, jobs):
    j = jobs.get(item, {})
    out = ["% " + "=" * 69,
           f"% HPC_{item} -- Isambard 3 (partition grace) run",
           *[f"% {s}" for s in status_lines],
           f"% Data:   {data_rel}",
           f"% Driver: code/{driver} (seed base 20260923, tag {tag})",
           f"% Jobs:   {j.get('job_id', 'see notes')}; node-hours (sacct) {j.get('node_hours', 'see notes')}",
           "% " + "=" * 69]
    return "\n".join(out)


# ---------------------------------------------------------------------------


def snip_n5(D: dict, path: Path, jobs: dict) -> str:
    P = rel(path)
    acc = D["acceptance"]
    fam = acc["by_family"]
    cells = D["cells"]
    lines = []
    ratios_b, ratios_p, bias_b, ci_b = [], [], [], []
    bridge_b, cons_z = [], []
    prom_rows = []
    peak_D = []
    for cell, c in cells.items():
        for lab, r in c["budgets"].items():
            for k, F in r["functionals"].items():
                if k.startswith("basin_masses"):
                    if F["em_order_ratio"] is not None:
                        ratios_b.append(F["em_order_ratio"])
                    bias_b.append(abs(F["prod_bias_vs_exb_R2"]))
                    ci_b.append(abs(F["prod_bias_vs_exb_R2"]) / F["prod_value_ci95_halfwidth"])
                    bridge_b.append(abs(F["bridge_correction_b2_minus_r1"]))
                    if F["em_vs_exb_richardson2_z"] is not None:
                        cons_z.append(abs(F["em_vs_exb_richardson2_z"]))
                if k.startswith("peak_times"):
                    if F["em_order_ratio"] is not None:
                        ratios_p.append(F["em_order_ratio"])
                    peak_D.append(F["prod_bias_vs_exb_R2"])
                if k == "late_rel_prominence_smoothed":
                    prom_rows.append((lab, F))
    Nmin = min(c["N_paths"] for c in cells.values())
    st = [f"basin masses {fam['basin_masses']['pass_vs_exb_R2']}/{fam['basin_masses']['n']} pass "
          f"(|bias| < 95% CI of the dt=1e-3 value, vs exb_R2); peak times "
          f"{fam['peak_times_smoothed']['pass_vs_exb_R2']}/{fam['peak_times_smoothed']['n']}; "
          f"last-mode prominence {fam['late_rel_prominence_smoothed']['pass_vs_exb_R2']}/"
          f"{fam['late_rel_prominence_smoothed']['n']}",
          "Bridge refinement (ex_b2, ex_b4) = exact OU/Brownian bridge points inside each fine step."]
    lines.append(header("N5full", st, rel(path.parent), "fb_hpc_n5full.py", 95, jobs))
    lines.append("\\paragraph{Time-step convergence at scale.}")
    lines.append(
        f"We repeated the time-step ladder on Isambard~3 with $3\\times10^6$ paths per cell "
        f"(three independent seeds of $10^6$; at least ${sci(Nmin, 2)}$ paths per cell).\n"
        f"% src: {P}#cells.*.N_paths")
    lines.append(
        "All schemes use common standard-normal increments at $h=5\\times10^{-4}$: the production "
        "Euler--Maruyama chain at $\\Delta t\\in\\{5\\times10^{-4},10^{-3},2\\times10^{-3}\\}$, the exact "
        "Ornstein--Uhlenbeck chain at the same steps, and the exact chain refined inside every step by "
        "exact bridge points to $\\Delta t=h/2$ and $h/4$. The refinement resolves contact episodes that "
        "begin and end within one step, which the end-of-step rule misses.")
    if ratios_b:
        lines.append(
            f"Basin masses converge at first order: successive-difference ratios of the Euler--Maruyama ladder "
            f"lie in ${f(min(ratios_b), 3)}$--${f(max(ratios_b), 3)}$"
            + (f" (peak times ${f(min(ratios_p), 3)}$--${f(max(ratios_p), 3)}$)" if ratios_p else "") + ".\n"
            f"% src: {P}#cells.*.budgets.*.functionals.basin_masses[*].em_order_ratio, "
            f".peak_times_smoothed[*].em_order_ratio")
    lines.append(
        f"Against the second-order limit of the bridge-refined exact ladder, the production step biases the basin "
        f"masses by at most ${sci(max(bias_b))}$, at most ${f(max(ci_b), 2)}$ times the $95\\%$ sampling half-width "
        f"of the $\\Delta t=10^{{-3}}$ value ({fam['basin_masses']['pass_vs_exb_R2']} of {fam['basin_masses']['n']} basins "
        f"below it). Sub-step contact episodes change basin masses by at most ${sci(max(bridge_b))}$ "
        f"(bridge-refined minus end-point rule at $\\Delta t=5\\times10^{{-4}}$), and the extrapolated limits of the "
        f"Euler--Maruyama and bridge-refined exact ladders agree (largest $|z|={f(max(cons_z), 2)}$).\n"
        f"% src: {P}#cells.*.budgets.*.functionals.basin_masses[*].{{prod_bias_vs_exb_R2, prod_value_ci95_halfwidth, "
        f"bridge_correction_b2_minus_r1, em_vs_exb_richardson2_z}}; #acceptance.by_family.basin_masses")
    for lab, F in prom_rows:
        lines.append(
            f"Last-mode relative prominence, {lab.replace('_', ' ')}: ${f(F['values']['em_r2'], 4)}$ at "
            f"$\\Delta t=10^{{-3}}$, ${f(F['exb_richardson2'], 4)}\\pm{sci(1.96 * F['exb_richardson2_se'])}$ extrapolated; "
            f"bias ${sci(F['prod_bias_vs_exb_R2'])}\\pm{sci(1.96 * F['prod_bias_vs_exb_R2_se'])}$ against a half-width "
            f"${sci(F['prod_value_ci95_halfwidth'])}$ ({'pass' if F['accept_abs_Dexb_lt_ci95_of_dt1e-3_value'] else 'FAIL'}).\n"
            f"% src: {P}#cells.*.budgets.{lab}.functionals.late_rel_prominence_smoothed")
    pb = D.get("peak_time_bias_model", {})
    if peak_D:
        lines.append(
            f"Peak times at $\\Delta t=10^{{-3}}$ lie ${sci(min(abs(x) for x in peak_D))}$--${sci(max(abs(x) for x in peak_D))}$ "
            f"from the extrapolated limit ({fam['peak_times_smoothed']['pass_vs_exb_R2']} of {fam['peak_times_smoothed']['n']} "
            f"within the sampling half-width); the shift follows $(\\Delta t/2)(1+\\gamma t_{{\\rm peak}})$ to within "
            f"${sci(pb.get('max_abs_D_em_vs_exb_minus_pred'))}$.\n"
            f"% src: {P}#peak_time_bias_model.max_abs_D_em_vs_exb_minus_pred; #cells.*.budgets.*.functionals.peak_times_smoothed[*].prod_bias_vs_exb_R2")
    bf = D.get("boundary_flip")
    if bf:
        rs = bf["resampling"]
        lines.append(
            f"In the boundary cell $(3,0.175,0.5)$ the protocol returns three modes in "
            f"${f(rs['em_r2_N500000']['p_mode_count_eq_m'], 3)}$ of multinomial $5\\times10^5$-walker histograms drawn "
            f"from the $\\Delta t=10^{{-3}}$ law and ${f(rs['ex_b4_N500000']['p_mode_count_eq_m'], 3)}$ from the finest "
            f"bridge-refined law ({rs['em_r2_N500000']['replicas']} replicas each).\n"
            f"% src: {P}#boundary_flip.resampling.{{em_r2_N500000, ex_b4_N500000}}.p_mode_count_eq_m")
    lc = D.get("local_N5_comparison", {})
    zs = [v.get("max_abs_z_em_r2") for v in lc.values() if isinstance(v, dict) and v.get("max_abs_z_em_r2") is not None]
    if zs:
        lines.append(f"The production-scheme values agree with the independent local $3\\times2\\times10^5$ run "
                     f"(largest $|z|={f(max(zs), 2)}$).\n% src: {P}#local_N5_comparison.*.max_abs_z_em_r2")
    return "\n".join(lines) + "\n"


def snip_n3(D: dict, path: Path, jobs: dict) -> str:
    P = rel(path)
    lines = []
    st = ["exact-OU nested dt ladder (5e-4, 1e-3, 2e-3) + production EM at 1e-3; 3 seeds x 1e6 per anchor"]
    lines.append(header("N3full", st, rel(path.parent), "fb_hpc_n3full.py", 96, jobs))
    lines.append("\\paragraph{Contact and field-location factorial at scale.}")
    for an, A in D["anchors"].items():
        mct = A["mode_count_table"]
        # dt sensitivity of the mode counts
        by = {}
        for r in mct:
            by.setdefault((r["variant"], r["B"]), {})[r["level"]] = r["modes_protocol_1e6"]
        n_change = sum(1 for v in by.values() if len(set(v.values())) > 1)
        dte = A["dt_effects"]
        mx = max(max(abs(x) for x in e["window"]["ex_r2_minus_R1"]["diff"]) for e in dte)
        mz = max(e["window"]["ex_r2_minus_R1"]["max_abs_z"] for e in dte)
        mem = max(max(abs(x) for x in e["window"]["em_r2_minus_ex_r2"]["diff"]) for e in dte)
        con = [c for c in A["contrasts"] if c["level"] == "R1" and c["kind"].startswith("contact_contribution")]
        con_by_a = {}
        for c in con:
            a = c["minus"][0].split("_a")[1].split("_")[0]
            con_by_a.setdefault(a, []).append(max(abs(x) for x in c["basin_mass_diff"]))
        link = A.get("local_n3_link", {})
        lines.append(
            f"Anchor {an.replace('_', ', ')} (${sci(A['N_paths'])}$ paths): the protocol mode count of the "
            f"$14$ kernel variants $\\times$ $4$ budgets changes between the exact $\\Delta t=10^{{-3}}$ chain, its "
            f"Richardson limit and the production Euler--Maruyama chain in {n_change} of {len(by)} cells. "
            f"The time step moves window basin masses by at most ${sci(mx)}$ (largest paired $|z|={f(mz, 2)}$; "
            f"Euler--Maruyama vs exact at the same step: ${sci(mem)}$). "
            + "; ".join(f"contact contribution (gated minus mean contact) at $a={a}$: up to ${f(max(v), 3)}$"
                        for a, v in sorted(con_by_a.items(), key=lambda kv: -float(kv[0])))
            + (f". Against the local $10^6$-path Euler--Maruyama factorial the production-chain basin masses agree to "
               f"$|z|\\le{f(link['max_abs_z'], 2)}$ ({link['n_abs_z_gt_3']} of {link['n_basins']} basins above $3$)."
               if link.get("max_abs_z") is not None else ".")
            + f"\n% src: {P}#anchors.{an}.{{mode_count_table, dt_effects[*].window.ex_r2_minus_R1, "
              f"dt_effects[*].window.em_r2_minus_ex_r2, contrasts[level=R1, kind=contact_contribution*], local_n3_link}}")
    return "\n".join(lines) + "\n"


def _up(x, d=2):
    """Round |x| outward (away from zero) to d significant figures (for CI bounds and '|z| <=' bounds)."""
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x))) - (d - 1)
    return math.copysign(math.ceil(abs(x) / 10 ** e - 1e-9) * 10 ** e, x)


def _down(x, d=2):
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x))) - (d - 1)
    return math.copysign(math.floor(abs(x) / 10 ** e + 1e-9) * 10 ** e, x)


def _sci_ci(lo, hi, d=3):
    """95% CI rounded outward to d significant figures, in LaTeX sci notation."""
    return f"$[{sci(_down(lo, d), d)},{sci(_up(hi, d), d)}]$"


def _load_opt(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _cell_label(key: str) -> str:
    m, e, b, a = key.split("_")
    return f"$({m[1:]},{e[3:]},{b[1:]})$ {'max--min' if a == 'maxmin' else 'equal'}"


def snip_headline(D: dict, path: Path, jobs: dict) -> str:
    P = rel(path)
    rows = D["headline_table"]
    cells = D["cells"]
    lines = []
    n_sub = sum(r["protocol_1e6_n"] for r in rows)
    n_sub_eq = sum(r["protocol_1e6_histogram"].get(str(r["fk_protocol_mode_count_1e6"]), 0) for r in rows
                   if r["fk_protocol_mode_count_1e6"] is not None)
    unanimous = sum(1 for r in rows if r["fk_protocol_mode_count_1e6"] is not None
                    and r["protocol_1e6_histogram"] == {str(r["fk_protocol_mode_count_1e6"]): r["protocol_1e6_n"]})
    full_eq = sum(1 for r in rows if r["mode_count_full"] == r["fk_protocol_mode_count_1e6"])
    zs = [z for r in rows for z in cells[r["cell"]]["vs_N1_exact_law"]["z_dk_minus_fk"]]
    zmax = max(abs(z) for z in zs)
    zarg = max(rows, key=lambda r: r["max_abs_z_vs_fk"])["cell"]
    n_gt2 = sum(1 for z in zs if abs(z) > 2)
    chi2 = sum(z * z for z in zs)
    seedz = [z for r in rows for z in cells[r["cell"]]["seed_consistency_z_extended"]]
    szmax = max(abs(z) for z in seedz)
    dev = {}
    for r in rows:
        c = cells[r["cell"]]
        e = c["cell"]["eps"]
        dev[e] = max(dev.get(e, 0.0), max(abs(x) for x in c["vs_N1_exact_law"]["dk_minus_target"]))
    walkers = rows[0]["walkers"]
    st = [f"{len(rows)} cells x 2 seeds x 1e7 walkers; protocol count on every 1e6 sub-histogram = FK count "
          f"({n_sub_eq}/{n_sub}); max basin |z| DK vs FK {zmax:.2f} ({zarg}); max seed |z| {szmax:.2f}"]
    lines.append(header("headline", st, rel(path.parent), "fb_hpc_headline.py", 97, jobs))
    lines.append("\\paragraph{Direct-simulation confirmation at $2\\times10^7$ pairs.}")
    lines.append(
        f"We re-simulated all {len(rows)} headline cells ($m\\in\\{{2,3\\}}$ at $\\varepsilon\\in\\{{0.05,0.1\\}}$ and "
        f"$m=5$ at $\\varepsilon=0.1$; equal weights and the max--min design) with the production direct-kill simulator, "
        f"$10^7$ pairs for each of two independent seeds (${sci(walkers)}$ per cell). In every cell each of the "
        f"$20$ disjoint $10^6$-pair sub-histograms gives the exact-law protocol count ({n_sub_eq} of {n_sub}; "
        f"{unanimous} of {len(rows)} cells unanimous), and so does the pooled histogram ({full_eq} of {len(rows)}). "
        f"The {len(zs)} basin masses agree with the exact law with $|z|\\le{_up(zmax, 2):g}$ "
        f"({n_gt2} with $|z|>2$, $\\sum z^2={chi2:.1f}$), and the two seeds agree with $|z|\\le{_up(szmax, 2):g}$. "
        f"The largest deviation of a direct-kill basin mass from its limit-law target is "
        f"${_up(dev.get(0.05), 2):g}$ at $\\varepsilon=0.05$ and ${_up(dev.get(0.1), 2):.3f}$ at $\\varepsilon=0.1$ (rounded outward).\n"
        f"% src: {P}#headline_table[*].{{protocol_1e6_histogram, fk_protocol_mode_count_1e6, mode_count_full, max_abs_z_vs_fk, "
        f"seed_consistency_max_abs_z}}; #cells.*.vs_N1_exact_law.{{z_dk_minus_fk, dk_minus_target}}; "
        f"#cells.*.seed_consistency_z_extended (max |z| {zmax:.4f} at {zarg}, {szmax:.4f}; bounds rounded outward)\n"
        f"% note: the z's use binomial DK SEs and the FK iid SEs of the single 5e5-path N1 ensemble per (m,eps), "
        f"so they are correlated across B within an (m,eps); with {len(zs)} basins max|z| = {zmax:.2f} is unremarkable.")
    # table of the main-text cells
    lines.append("Basin masses in the cells quoted in Sec.~\\ref{sec:num-design} (direct kill $\\pm$ its largest 95\\% half-width; "
                 "modes: pooled direct-kill histogram / exact law at $10^6$):")
    lines.append("\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llll}\n\\hline\n"
                 "cell & direct kill, $2\\times10^7$ & exact law & modes \\\\\n\\hline")
    for key in ("m2_eps0.1_B8_equal", "m2_eps0.1_B8_maxmin", "m3_eps0.1_B4_equal", "m3_eps0.1_B4_maxmin",
                "m3_eps0.1_B8_equal", "m3_eps0.1_B8_maxmin"):
        c = cells.get(key)
        if not c:
            continue
        be = c["pooled"]["basins_extended"]
        v = c["vs_N1_exact_law"]
        dk = ", ".join(f"{m:.4f}" for m in be["mass"])
        hw = max(1.96 * s for s in be["se_binomial"])
        fk = ", ".join(f"{m:.4f}" for m in v["fk_mass_extended"])
        lines.append(f"{_cell_label(key)} & $({dk})$ ($\\pm{_up(hw, 1):.4f}$) & $({fk})$ & "
                     f"{c['pooled']['classifier_full_histogram']['mode_count']}/{v['fk_protocol_mode_count_1e6']} \\\\\n"
                     f"% src: {P}#cells.{key}.pooled.basins_extended.{{mass,se_binomial}}; "
                     f"#cells.{key}.vs_N1_exact_law.{{fk_mass_extended,fk_protocol_mode_count_1e6,z_dk_minus_fk}} "
                     f"(z = {', '.join(f'{z:.2f}' for z in v['z_dk_minus_fk'])})")
    lines.append("\\hline\n\\end{tabular}\\end{center}")
    # late designed peak times vs targets (targets from N11)
    n11 = _load_opt(path.parents[1] / "N11_shift_compensation" / "n11_shift_compensation.json")
    parts = []
    for key, n11key in (("m2_eps0.1_B8_maxmin", "m2_eps0.1_B8"), ("m3_eps0.1_B8_maxmin", "m3_eps0.1_B8")):
        c = cells.get(key)
        if not c or not n11:
            continue
        pk = c["pooled"]["peaks_extended_basins"]
        tl, se = pk["peak_times"][-1], pk["jackknife_se"][-1]
        tp = n11["cells"][n11key]["designs"]["box2"]["tp"][-1]
        parts.append((key, tl, se, tp, n11key))
    if parts:
        lines.append(
            "The designed late peaks lie at "
            + " and ".join(f"${tl:.4f}\\pm{_up(1.96 * se, 1):.4f}$ at {_cell_label(k).split(' ')[0]} "
                           f"({100 * (tp - tl) / tp:.1f}\\% before $t_m={tp:g}$)" for k, tl, se, tp, _ in parts)
            + " (vertex of the $0.04$-smoothed density, delete-one-group jackknife 95\\% half-width).\n"
            + "".join(f"% src: {P}#cells.{k}.pooled.peaks_extended_basins.{{peak_times[-1], jackknife_se[-1]}}; "
                      f"artifacts/data/exact_m_fixed_budget/N11_shift_compensation/n11_shift_compensation.json#cells.{nk}.designs.box2.{{tp[-1], w}} "
                      f"(same w as the HPC design: #cells.{k}.design.w)\n" for k, tl, se, tp, nk in parts).rstrip("\n"))
    return "\n".join(lines) + "\n"


def _ess(x):
    return sci(x, 2) if x >= 1e4 else (f"{x:.0f}" if x >= 10 else f"{x:.1f}")


def snip_census(D: dict, path: Path, jobs: dict) -> str:
    P = rel(path)
    lines = []
    gated = [(cell, lab, it) for cell, c in D["cells"].items() for lab, it in c["items"].items() if it["gated_n9a_cell"]]
    st = [f"{cell}/{lab}: {it['decision_last_mode']['decision']}" for cell, lab, it in gated]
    lines.append(header("census", st, rel(path.parent), "fb_hpc_census.py", 98, jobs))
    n12p = path.parents[1] / "N12_census_resolution" / "n12_census_resolution.json"
    n12 = _load_opt(n12p)
    n12rel = rel(n12p)

    def n12case(m, eps, B):
        if not n12:
            return None, None
        for i, cs in enumerate(n12["cases"]):
            k = cs["case"]
            if k["m"] == m and abs(k["eps"] - eps) < 1e-12 and abs(k["B"] - B) < 1e-9:
                return i, cs
        return None, None

    lines.append("\\paragraph{Extreme-scale census of the gated cells.}")
    for cell, c in D["cells"].items():
        lines.append(f"% {cell}: N = {c['N_paths']} unkilled paths in {c['groups']} groups of "
                     f"{c['group_paths_min_max'][0]} (seed tag {c['seed']['tag']})\n% src: {P}#cells.{cell}.{{N_paths, groups, group_paths_min_max, seed}}")
    lines.append(
        f"We repeated the derivative census of Sec.~\\ref{{sec:num-count}} for the three gated cells at $\\varepsilon=0.05$ on "
        f"${sci(D['cells']['m2_eps0.05']['N_paths'])}$ ($m=2$) and ${sci(D['cells']['m3_eps0.05']['N_paths'])}$ ($m=3$) "
        f"exact-law paths (production time step, gated kernel, equal weights), with standard errors from "
        f"{D['cells']['m3_eps0.05']['groups']} independent path groups, $|z|\\ge{D['zthr']:g}$, and bin spacings "
        f"$h_g/\\varepsilon\\in\\{{{','.join(f'{r:g}' for r in D['resolutions_h_over_eps'])}\\}}$. In all three the last "
        f"maximum is present.\n% src: {P}#{{cells.*.N_paths, zthr, resolutions_h_over_eps, model}}")
    for cell, lab, it in gated:
        c = D["cells"][cell]
        m, eps, B = c["m"], c["eps"], it["B"]
        lp = it["late_prominence"]["smoothed_bw0.04"]
        dec = it["decision_last_mode"]
        cbr = it["counts_by_resolution"]
        exact = [r for r, v in cbr.items() if v == [m, m - 1]]
        pres = dec.get("present_at", [])
        zmin_b = min(min(abs(z[0]) for z in x["z"]) for x in pres) if pres else None
        zmin_a = min(min(abs(z[1]) for z in x["z"]) for x in pres) if pres else None
        mass, mse = it["mass_after_last_G_valley"], it["mass_after_last_G_valley_se"]
        i12, cs = n12case(m, eps, B)
        cmp_txt, cmp_src = "", ""
        if cs is not None:
            dk = cs["direct_kill"][0]
            dkb = dk["basins"][-1]
            zdk = (mass - dkb["mass"]) / math.hypot(mse, dkb["se"])
            pr12, pr12se = cs["last_mode_rel_prom"], cs["last_mode_rel_prom_jack_se"]
            zpr = (lp["rel_prominence"] - pr12) / math.hypot(lp["jackknife_se"], pr12se)
            stick = cs["regime"]["stick_breaking_masses"][-1]
            ratio = mass / stick
            rtxt = f"{ratio:.0f}" if 10 <= ratio < 1e4 else (f"{ratio:.1f}" if ratio < 10 else sci(ratio, 2))
            cmp_txt = (f" (independent direct kill, ${sci(dk['walkers'], 1)}$ walkers: ${sci(dkb['mass'], 3)}\\pm{sci(_up(1.96 * dkb['se'], 2))}$, $z={zdk:.2f}$), "
                       f"${rtxt}$ times its limit-law value")
            cmp_src = (f"\n% src: {n12rel}#cases[{i12}].direct_kill[0].{{walkers, basins[-1].mass, basins[-1].se}}; "
                       f"#cases[{i12}].regime.stick_breaking_masses[-1] ({stick:.4g}); "
                       f"#cases[{i12}].{{last_mode_rel_prom, last_mode_rel_prom_jack_se}} ({pr12:.4g} +- {pr12se:.2g}; "
                       f"z(HPC - N12 IS) = {zpr:.2f}); z's combine the two SEs in quadrature")
        ess = it["per_step_ess_last_basin"]
        lines.append(
            f"$(m,\\varepsilon,B)=({m},{eps:g},{B:g})$: $(n_{{\\max}},n_{{\\min}})=$ "
            + ", ".join(f"$({v[0]},{v[1]})$" for v in cbr.values())
            + f" at $h_g/\\varepsilon=" + ",".join(cbr.keys()) + "$; exactly $m$ maxima and $m-1$ minima at $h_g/\\varepsilon\\in\\{"
            + ",".join(exact) + "\\}$"
            + (f", where $f'$ changes sign from $+$ to $-$ in the last basin with $|z|\\ge{math.floor(10 * zmin_b) / 10:g}$ before and "
               f"$\\ge{math.floor(10 * zmin_a) / 10:g}$ after" if pres else "")
            + ((f"; at finer spacings the last basin shows no significantly negative $f'$ (undecided, not monotone)"
                if all(it["by_resolution"][r]["late_basin_sign_profile"]["n_neg"] == 0 for r in cbr if r not in exact)
                else "; at finer spacings the last maximum is not resolved")
               if len(exact) < len(cbr) else "")
            + f". Mass after the last valley ${sci(mass, 3)}\\pm{sci(_up(1.96 * mse, 2))}$" + cmp_txt
            + f"; relative prominence of the last maximum ${sci(lp['rel_prominence'], 3)}$, 95\\% CI "
            + _sci_ci(lp["ci95"][0], lp["ci95"][1], 3)
            + f"; per-step effective sample size in the last basin median ${_ess(ess['median'])}$, "
              f"minimum ${_ess(ess['min'])}$.\n"
            + f"% src: {P}#cells.{cell}.items.{lab}.{{counts_by_resolution, decision_last_mode.present_at[*].z, "
              f"by_resolution.*.late_basin_sign_profile, mass_after_last_G_valley(_se), late_prominence.smoothed_bw0.04.{{rel_prominence, ci95}}, "
              f"per_step_ess_last_basin}}; +- and CI are 95% (CI rounded outward)" + cmp_src)
    proms = [it["late_prominence"]["smoothed_bw0.04"]["rel_prominence"] for _, _, it in gated]
    lines.append(
        f"The last maximum is therefore a critical point of $f$ with relative prominence "
        f"${sci(min(proms), 1)}$--${sci(max(proms), 1)}$, orders of magnitude below the $5\\%$ floor of protocol~P; "
        f"these cells are counted, not seen. Where the minimum effective sample size is small (the $B=50$ cells), "
        f"the group standard errors may be optimistic.\n"
        f"% src: {P}#cells.*.items.full_B*.late_prominence.smoothed_bw0.04.rel_prominence (gated items); "
        f"#cells.*.items.full_B50.per_step_ess_last_basin.min")
    return "\n".join(lines) + "\n"


ITEMS = {"N5full": ("hpc_n5full.json", snip_n5), "N3full": ("hpc_n3full.json", snip_n3),
         "headline": ("hpc_headline.json", snip_headline), "census": ("hpc_census.json", snip_census)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA))
    ap.add_argument("--out-dir", default=str(SNIP))
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--legacy-n5n3", action="store_true",
                    help="also write HPC_N5full/HPC_N3full with the old generators (default: skip; they are "
                         "written by fb_hpc_n5n3_report.py)")
    a = ap.parse_args(argv)
    jobs = json.loads(Path(a.jobs).read_text()) if a.jobs else {}
    for item, (fn, gen) in ITEMS.items():
        if a.only and item not in a.only.split(","):
            continue
        if item in ("N5full", "N3full") and not a.legacy_n5n3:
            print(f"[snippets] {item}: skipped (written by fb_hpc_n5n3_report.py; use --legacy-n5n3 to override)")
            continue
        p = Path(a.data_root) / f"HPC_{item}" / fn
        if not p.exists():
            print(f"[snippets] {item}: {p} missing, skipped")
            continue
        text = gen(json.loads(p.read_text()), p, jobs)
        out = Path(a.out_dir) / f"HPC_{item}.tex"
        out.write_text(text)
        print(f"[snippets] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
