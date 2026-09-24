"""W6 single-particle width/shift diagnostic (2026-09-23, S1 audit follow-up; reads stored JSONs only).
Width ratio single/pair with each histogram's log-parabola fit centred on ITS OWN vertex
(iterated), window half-width hw = sigma_x/|mu'(t)| as in s1_widths2.py. Poisson bootstrap SE
(rng seed 20260923, tag 87 -> SeedSequence([20260923, 87]))."""
import json, math, numpy as np
R='artifacts/data/'
D=R+'exact_m_prr_upgrade/w6_single_particle/'
rng=np.random.default_rng(np.random.SeedSequence([20260923,87]))
def fit(t,c,tp,hw):
    m=(t>tp-hw)&(t<tp+hw)&(c>0); x=t[m]-tp; y=np.log(c[m])
    a,b,_=np.polyfit(x,y,2,w=np.sqrt(c[m])); return math.sqrt(-1/(2*a)), tp-b/(2*a)
def selfc(t,c,tp,hw):
    for _ in range(6):
        s,v=fit(t,c,tp,hw); tp=v
    return s,tp
out={}
for f in ['m2_eps0.05_B1_p1.json','m2_eps0.1_B1_p1.json','m3_eps0.1_B1_p1.json','m2_eps0.05_B1_p2.json','m2_eps0.1_B1_p2.json','m3_eps0.1_B1_p2.json']:
    r=json.load(open(D+f)); eps=r['parameters']['config']['eps']
    comp=json.load(open(R+r['parameters']['comparator']))
    sigx=eps*math.sqrt(1.5); rows=[]
    for tp0 in [p['time'] for p in r['theory']['single_particle_G1']['peaks']]:
        hw=sigx/(4*math.exp(-tp0)); res=[]
        for cl in (r['results']['classifier'], comp['results']['classifier']):
            e=np.array(cl['edges']); c=np.array(cl['counts'],float); tc=0.5*(e[1:]+e[:-1])
            s0,v0=selfc(tc,c,tp0,hw); bs=[selfc(tc,rng.poisson(c).astype(float),tp0,hw) for _ in range(200)]
            res.append((s0,float(np.std([b[0] for b in bs])),v0,float(np.std([b[1] for b in bs]))))
        ratio=res[0][0]/res[1][0]; se=ratio*math.hypot(res[0][1]/res[0][0],res[1][1]/res[1][0])
        rows.append(dict(t_G1=tp0,hw=hw,sigma_single=res[0][0],se_single=res[0][1],sigma_pair=res[1][0],se_pair=res[1][1],
                         ratio=ratio,se_ratio=se,vertex_single=res[0][2],vertex_pair=res[1][2],
                         vertex_shift=res[0][2]-res[1][2],se_shift=math.hypot(res[0][3],res[1][3])))
        print(f, f"t~{tp0:.3f} ratio={ratio:.3f}+-{se:.3f} vertex shift={res[0][2]-res[1][2]:+.4f}+-{math.hypot(res[0][3],res[1][3]):.4f}")
    out[f]=rows
json.dump({"script":"code/exact_m_prr_w6_width_diagnostic.py","seed":[20260923,87],"method":__doc__,"cells":out},open(D+'width_diagnostic_selfcentred.json','w'),indent=1)
