import sys,math,json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
P=json.load(open(R/'config/twin_params.json')); dt=P['ts']
S=R/'data/extracted/2026-05-18_19-2-15/splits'
def series(ev):
    c=pd.read_csv(ev/'commands.csv'); r=pd.read_csv(ev/'processed_robots.csv')
    t0=c['timestamp'].iloc[0]
    tg=np.arange(0,min(c['timestamp'].iloc[-1],r['timestamp'].iloc[-1])-t0,dt)
    if len(tg)<60: return None
    u=np.interp(tg,c['timestamp']-t0,c['move_w'])
    th=np.unwrap(r['position_w'].to_numpy())
    y=np.gradient(savgol_filter(np.interp(tg,r['timestamp']-t0,th),7,2))/dt
    return u,y
def sim(u,K,tau,Td):
    m=int(math.ceil(max(Td,0)/dt)); a=1-math.exp(-dt/max(tau,1e-4))
    v=0.0; out=np.empty(len(u))
    for k in range(len(u)):
        ud=u[k-m] if k-m>=0 else 0.0
        v=v+a*(ud-v); out[k]=K*v
    return out
def fitpct(y,yh): return 100*(1-np.linalg.norm(y-yh)/np.linalg.norm(y-np.mean(y)))
# eventos de malha fechada mais ricos em rotacao
evs=[e for e in sorted((S/'shoot_to_goal').iterdir()) if e.is_dir()]
D=[(e.name,)+tuple(series(e)) for e in evs if series(e) is not None]
# mesmo criterio da linha open-loop: split event-wise, 2/3 treino 1/3 hold-out
rng=np.random.default_rng(42); idx=rng.permutation(len(D))
nho=max(1,round(len(D)/3)); ho=sorted(idx[:nho]); tr=sorted(idx[nho:])
def cost(p,ids):
    K,tau,Td=p; s=0.0
    for i in ids:
        _,u,y=D[i]; s+=np.sum((y-sim(u,K,tau,Td))**2)
    return s
res=minimize(cost,[1.3,0.03,0.10],args=(tr,),method='Nelder-Mead',
             options=dict(maxiter=400,xatol=1e-4,fatol=1e-3))
K,tau,Td=res.x
f_tr=[fitpct(D[i][2],sim(D[i][1],K,tau,Td)) for i in tr]
f_ho=[fitpct(D[i][2],sim(D[i][1],K,tau,Td)) for i in ho]
ts=datetime.now().strftime('%Y%m%d_%H%M')
L=[f"# R3 - fit hold-out do ajuste de omega em MALHA FECHADA - {ts}","",
 "Mesmo criterio da linha open-loop (Eq. 2 do artigo): split event-wise, media",
 "sobre os eventos de hold-out. Fecha a lacuna que deixava a linha closed-loop",
 "sem um numero comparavel (objecao C5 da 1a revisao independente).","",
 f"- Eventos `shoot_to_goal` usados: {len(D)} ({len(tr)} treino / {len(ho)} hold-out)",
 f"- Parametros: K = {K:.4f}, tau = {tau:.4f} s, Td = {Td:.4f} s",
 f"- **fit hold-out = {np.mean(f_ho):.1f} %** (por evento: {', '.join(f'{v:.1f}' for v in f_ho)})",
 f"- fit treino (mediana) = {np.median(f_tr):.1f} %","",
 f"Comparacao direta, agora sob o MESMO criterio:","",
 "| regime | K | tau (s) | Td (s) | fit hold-out |","|---|---|---|---|---|",
 f"| closed-loop (competicao) | {K:.3f} | {tau:.4f} | {Td:.4f} | **{np.mean(f_ho):.1f} %** |",
 f"| open-loop (dedicado) | 1.002 | 0.0828 | 0.0890 | **71.1 %** |","",
 "Gate pre-registrado: fit hold-out >= 60 %."]
o=R/'results'/f'r3_closedloop_omega_fit_{ts}.md'; o.write_text("\n".join(L),encoding='utf-8')
print("\n".join(L))
