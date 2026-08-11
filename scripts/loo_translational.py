import sys,math,json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
P=json.load(open(R/'config/twin_params.json')); dt=P['ts']
S=R/'data/extracted/2026-05-18_19-2-15/splits'
LB={'vx':('front_to_back','move_x'),'vy':('side_to_side','move_y')}
def series(ev,col,dof):
    c=pd.read_csv(ev/'commands.csv'); r=pd.read_csv(ev/'processed_robots.csv')
    t0=c['timestamp'].iloc[0]
    tg=np.arange(0,min(c['timestamp'].iloc[-1],r['timestamp'].iloc[-1])-t0,dt)
    if len(tg)<60: return None
    u=np.interp(tg,c['timestamp']-t0,c[col])
    tr=r['timestamp']-t0
    th=np.interp(tg,tr,np.unwrap(r['position_w'].to_numpy()))
    x=np.interp(tg,tr,r['position_x'].to_numpy())/1000.0; y=np.interp(tg,tr,r['position_y'].to_numpy())/1000.0  # mm -> m
    vx=np.gradient(savgol_filter(x,7,2))/dt; vy=np.gradient(savgol_filter(y,7,2))/dt
    cs,sn=np.cos(th),np.sin(th)
    yb=(cs*vx+sn*vy) if dof=='vx' else (-sn*vx+cs*vy)
    return u,yb
def sim(u,K,tau,Td):
    tau=max(tau,1e-3); Td=max(Td,0.0)
    m=int(math.ceil(max(Td,0)/dt)); a=1-math.exp(-dt/max(tau,1e-4)); v=0.0
    o=np.empty(len(u))
    for k in range(len(u)):
        ud=u[k-m] if k-m>=0 else 0.0
        v+=a*(ud-v); o[k]=K*v
    return o
def fitpct(y,yh): return 100*(1-np.linalg.norm(y-yh)/np.linalg.norm(y-np.mean(y)))
out=[f"# Leave-one-event-out por DOF translacional - {datetime.now():%Y%m%d_%H%M}","",
 "Refit completo a cada dobra, mantendo a estrutura P1D e a metrica Eq.(2).",
 "NAO altera o modelo promovido em config/twin_params.json -- e analise de",
 "dispersao, nao re-identificacao. Responde a objecao recorrente de que o fit",
 "de vx repousa sobre UM unico evento de hold-out.",""]
for dof,(lab,col) in LB.items():
    D=[]
    for ev in sorted((S/lab).iterdir()):
        if not ev.is_dir(): continue
        s=series(ev,col,dof)
        if s: D.append((ev.name,)+s)
    fits=[]
    for i in range(len(D)):
        tr=[j for j in range(len(D)) if j!=i]
        f=lambda q: sum(np.sum((D[j][2]-sim(D[j][1],q[0],math.exp(q[1]),math.exp(q[2])))**2) for j in tr)
        r=minimize(f,[1.1,math.log(0.08),math.log(0.06)],method='Nelder-Mead',
                   options=dict(maxiter=350,xatol=1e-4,fatol=1e-3))
        pk=(r.x[0],math.exp(r.x[1]),math.exp(r.x[2]))
        fits.append((D[i][0],fitpct(D[i][2],sim(D[i][1],*pk)),*pk))
    v=np.array([x[1] for x in fits]); K=np.array([x[2] for x in fits])
    out+=[f"## {dof} (`{lab}`, {len(D)} eventos, {len(D)} dobras)","",
          "| evento retido | fit (%) | K | tau (s) | Td (s) |","|---|---|---|---|---|"]
    for n,fp,k_,t_,d_ in fits: out.append(f"| {n} | {fp:.1f} | {k_:.3f} | {t_:.4f} | {d_:.4f} |")
    out+=["",f"- fit medio **{v.mean():.1f} %**, desvio **{v.std(ddof=1):.1f} p.p.**, "
          f"faixa {v.min():.1f}-{v.max():.1f} %",
          f"- K medio {K.mean():.3f}, desvio {K.std(ddof=1):.3f} "
          f"(promovido: {P['dof'][dof]['K']:.3f})",""]
o=R/'results'/f'loo_translational_{datetime.now():%Y%m%d_%H%M}.md'
o.write_text("\n".join(out),encoding='utf-8'); print("\n".join(out))
