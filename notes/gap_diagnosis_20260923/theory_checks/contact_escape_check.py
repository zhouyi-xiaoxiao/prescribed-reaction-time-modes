# Operator check: does contact gating cap the slab-1 kill at large B (full model) vs contact-one control?
import sys, math, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, '<local-ensemble-store>/codex_theory_prr_20260923/encounter_multimodal_prr/code')
import numpy as np
import exact_m_prr_upgrade_core as c
from dataclasses import replace
out = {}
for eps in (0.1, 0.05):
  for label, p in (('full', c.MODEL), ('contact_one', replace(c.MODEL, contact_a=10.0))):
    for B in (1.0, 10.0, 100.0, 1000.0, 10000.0):
      rng = np.random.default_rng(12345)
      n = 40000 if eps == 0.1 else 40000
      dt = 2e-4 if B <= 100 else 5e-5
      steps = int(round(4.5 / dt))
      t0 = time.time()
      r = c.simulate_chunk_general(rng, n, eps=eps, budget=B, weights=(0.5, 0.5),
                                   centres_z=c.centres_z_for(2), dt=dt, step_count=steps, p=p)
      kt = r['kill_times']
      m1 = float(np.mean(kt < 1.75)) * len(kt) / n
      m2 = float(np.sum(kt >= 1.75)) / n
      h, e = np.histogram(kt[(kt >= 0.5) & (kt <= 3.5)], bins=60, range=(0.5, 3.5))
      key = f"eps{eps}_{label}_B{B:g}"
      out[key] = dict(M1_window=m1, M2_window=m2, survivors=r['survivors'] / n,
                      hist=h.tolist(), secs=round(time.time() - t0, 1),
                      beta1=B * 0.5 / 1.4715177646857693, beta2=B * 0.5 / 0.3283399944955952)
      print(key, f"M1={m1:.4f} M2={m2:.4f} surv={r['survivors']/n:.4f} secs={time.time()-t0:.1f}", flush=True)
json.dump(out, open('<scratch>/prr_gap/theory_checks/contact_escape_check.json', 'w'), indent=1)
