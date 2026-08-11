import sys,json,math
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
P=json.load(open(R/'config/twin_params.json')); dt=P['ts']
S=R/'data/extracted/2026-05-18_19-2-15/splits'
LB={'vx':'front_to_back','vy':'side_to_side','w':'shoot_to_goal'}
COL={'vx':'move_x','vy':'move_y','w':'move_w'}
def load(ev):
    c=pd.read_csv(ev/'commands.csv'); r=pd.read_csv(ev/'processed_robots.csv')
    return c,r
def series(ev,dof):
    c,r=load(ev); t0=c['timestamp'].iloc[0]
    tg=np.arange(0,min(c['timestamp'].iloc[-1],r['timestamp'].iloc[-1])-t0,dt)
    if len(tg)<40: return None
    u=np.interp(tg,c['timestamp']-t0,c[COL[dof]])
    th=np.unwrap(r['position_w'].to_numpy()); px=r['position_x'].to_numpy(); py=r['position_y'].to_numpy()
    tr=r['timestamp']-t0
    thg=np.interp(tg,tr,th)
    if dof=='w':
        y=np.gradient(savgol_filter(thg,7,2))/dt
    else:
        xg=np.interp(tg,tr,px); yg=np.interp(tg,tr,py)
        vx=np.gradient(savgol_filter(xg,7,2))/dt; vy=np.gradient(savgol_filter(yg,7,2))/dt
        cs,sn=np.cos(thg),np.sin(thg)
        y=(cs*vx+sn*vy) if dof=='vx' else (-sn*vx+cs*vy)
    return u,y
rows=[]
for dof in ('vx','vy','w'):
    Td=P['dof'][dof]['Td']; m=int(round(Td/dt))
    U=[];Y=[]
    for ev in sorted((S/LB[dof]).iterdir()):
        if not ev.is_dir(): continue
        s=series(ev,dof)
        if s is None: continue
        u,y=s
        if m>0: u=u[:-m]; y=y[m:]
        U.append(u); Y.append(y)
    u=np.concatenate(U); y=np.concatenate(Y)
    # correlacao cruzada no lag do Td (o "0.99" reportado no artigo)
    r=float(np.corrcoef(u,y)[0,1])
    # fracao da variancia de y explicada por u (R^2 de regressao linear simples)
    b=np.polyfit(u,y,1); yh=np.polyval(b,u)
    r2=1-float(np.sum((y-yh)**2)/np.sum((y-np.mean(y))**2))
    rows.append(dict(dof=dof,label=LB[dof],n=len(u),corr=r,R2=r2,
                     std_cmd=float(np.std(u)),std_meas=float(np.std(y))))
ts=datetime.now().strftime('%Y%m%d_%H%M')
L=[f"# Excitacao por DOF nos logs de competicao - {ts}","",
 "Comando alinhado pelo Td identificado; velocidade de corpo medida pelo mesmo",
 "pre-processamento do artigo (Savitzky-Golay janela 7 ordem 2 + diferencas centrais).","",
 "| DOF | label | n | corr(cmd, meas) | R^2 (var. explicada) | std cmd | std meas |",
 "|---|---|---|---|---|---|---|"]
for d in rows:
    L.append(f"| {d['dof']} | `{d['label']}` | {d['n']} | **{d['corr']:.3f}** | **{d['R2']:.3f}** "
             f"| {d['std_cmd']:.3f} | {d['std_meas']:.3f} |")
L+=["","Leitura: para vx/vy o comando explica a maior parte da variancia da velocidade",
 "medida; para omega, nao. Esse e o diagnostico espectral/de correlacao que faltava",
 "(objecoes C8/C9 e perguntas 7-8 do revisor)."]
o=R/'results'/f'excitation_per_dof_{ts}.md'; o.write_text("\n".join(L),encoding='utf-8')
print("\n".join(L))
