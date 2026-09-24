import numpy as np, math, json, sys
sys.path.insert(0,'.')
from analyze_fk import count_modes, limit_masses, exact_basin_masses, meanfield_basin_masses
out={}
# orthant prediction: eta_perp = N(0, 0.09) + 2 BM -> var_j = 0.09 + 4 t_j
v1, v2 = 0.09+4*1.0, 0.09+4*2.5
rho = math.sqrt(v1/v2)
P11 = 0.25 + math.asin(rho)/(2*math.pi); P01 = 0.5 - P11
print('orthant: rho12=%.4f P11=%.4f P01=%.4f'%(rho,P11,P01))
for eps in (0.05,0.025):
    d=np.load(f'fkh3_m2_e{eps}.npz')
    chi=d['chi_tj'].astype(float)
    emp=dict(P1=float(chi[:,0].mean()),P2=float(chi[:,1].mean()),P11=float((chi[:,0]*chi[:,1]).mean()),P01=float(((1-chi[:,0])*chi[:,1]).mean()))
    print('eps',eps,'empirical gate stats',{k:round(v,4) for k,v in emp.items()})
    rows=[]
    for B in (0.5,1,2,4,8,20,50,200):
        lam=np.array([B*0.5/(4*math.exp(-1.0)), B*0.5/(4*math.exp(-2.5))])
        orth=[0.5*(1-math.exp(-lam[0])), (P11*math.exp(-lam[0])+P01)*(1-math.exp(-lam[1]))]
        _,pemp=limit_masses(2,B,chi=chi)
        ex,_=exact_basin_masses(d['lamck_full'],B)
        mf=meanfield_basin_masses(d['lamck_full'],B)
        rows.append(dict(B=B,exact=ex.tolist(),orthant=orth,frozen_emp=pemp.tolist(),mf=mf.tolist()))
        print('  B',B,'exact',np.round(ex,4),'orthant',np.round(orth,4),'frozen(emp chi)',np.round(pemp,4),'MF',np.round(mf,4))
    t=d['tgrid']; dt=float(d['dt']); budgets=d['budgets']
    FK=d['FK_full']/dt; FKb=d['FKb_full']/dt
    for bi,b in enumerate(budgets):
        cm=count_modes(t,FK[bi],FKb[:,bi,:],h_steps=max(1.0,0.15*eps*1.2247/1.4715/dt))
        sig=[(round(x['t'],3),'%.2e'%x['rel_prom'],round(x['z'],1)) for x in cm if x['z']>5]
        print('  modes B',b,'n_sig',len(sig),sig[:4],'n_all',len(cm))
    out[str(eps)]=dict(gate=emp,rows=rows)
json.dump(out,open('analysis_h3.json','w'),indent=1)
