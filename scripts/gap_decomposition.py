import sys,math,json,pickle
from pathlib import Path
from datetime import datetime
import numpy as np
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
from phase0_replay.splits import load_holdout_segments
from phase0_replay.params import load_twin_params
from phase0_replay.metrics import load_metrics_config, evaluate
from phase0_replay.plant import AnalyticFOPDTPlant, IdealKinematicPlant
from phase0_replay.rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
LW=15
p=load_twin_params(); mcfg=load_metrics_config(); H=list(mcfg['horizons'])
segs,_=load_holdout_segments(min_displacement_m=0.1)
class DP:  # DofParam simples
    def __init__(s,K,tau,Td): s.K,s.tau,s.Td=K,tau,Td; s.placeholder=False
def variant(useK,useTau,useTd):
    return {d:DP(p[d].K if useK else 1.0, p[d].tau if useTau else 0.0,
                 p[d].Td if useTd else 0.0) for d in ('vx','vy','w')}
VAR=[('ideal (K=1, tau=0, Td=0)',variant(0,0,0)),
     ('+ gain K',               variant(1,0,0)),
     ('+ time constant tau',    variant(1,1,0)),
     ('full twin (+ dead time)',variant(1,1,1))]
def ros(params): return [rollout_segment(s,AnalyticFOPDTPlant(params),h_max=DEFAULT_HMAX,
        stride=DEFAULT_STRIDE,lookback=LW,horizons=H) for s in segs]
base=ros(VAR[0][1])
print(f"{'variant':26} " + " ".join(f"h={h/60:.2f}" for h in H))
rows=[]
for name,par in VAR:
    r=evaluate(ros(par),base,params=p,mcfg=mcfg)['scopes']['trans']
    med=r['epos_twin_med']; rows.append((name,med))
    print(f"{name:26} " + " ".join(f"{m:6.4f}" for m in med))
ts=datetime.now().strftime('%Y%m%d_%H%M')
L=["# Decomposicao do gap - "+ts,"",
   "Mediana de e_pos (m) no bin translacional, mesmas janelas/horizontes.","",
   "| variante | "+" | ".join(f"h={h/60:.2f}s" for h in H)+" |",
   "|---|"+"---|"*len(H)]
for n,m in rows: L.append(f"| {n} | "+" | ".join(f"{v:.4f}" for v in m)+" |")
L+=["","## Contribuicao incremental (reducao de e_pos vs a variante anterior)",
    "| passo | "+" | ".join(f"h={h/60:.2f}s" for h in H)+" |","|---|"+"---|"*len(H)]
for i in range(1,len(rows)):
    d=[rows[i-1][1][k]-rows[i][1][k] for k in range(len(H))]
    L.append(f"| {rows[i][0]} | "+" | ".join(f"{v:+.4f}" for v in d)+" |")
o=R/'results'/f'gap_decomposition_{ts}.md'; o.write_text("\n".join(L),encoding='utf-8')
print("\n"+"\n".join(L[8:]))
print("\nartefato:",o.name)
