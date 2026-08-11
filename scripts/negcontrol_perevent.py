import sys, h5py, numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import wilcoxon
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
from phase0_replay.splits import load_holdout_segments
from phase0_replay.params import load_twin_params
from phase0_replay.metrics import load_metrics_config, build_error_table
from phase0_replay.plant import AnalyticFOPDTPlant, IdealKinematicPlant
from phase0_replay.rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
LW=15; P=load_twin_params(); mcfg=load_metrics_config(); H=list(mcfg['horizons'])
segs,_=load_holdout_segments(min_displacement_m=0.1)
with h5py.File(R/'models/sysid_bayesopt_v2_candidate.mat','r') as f:
    v=np.array(f['theta_opt']).squeeze()
    T=dict(zip(['K_vx','tau_vx','Td_vx','K_vy','tau_vy','Td_vy','K_w','tau_w','Td_w'],
               [float(x) for x in v]))
print('theta bayesopt:',{k:round(v,4) for k,v in T.items()})
class DP:
    def __init__(s,K,t,d): s.K,s.tau,s.Td=K,t,d; s.placeholder=False
BO={'vx':DP(T['K_vx'],T['tau_vx'],T['Td_vx']),'vy':DP(T['K_vy'],T['tau_vy'],T['Td_vy']),
    'w':DP(T['K_w'],T['tau_w'],T['Td_w'])}
def ros(f): return [rollout_segment(s,f(),h_max=DEFAULT_HMAX,stride=DEFAULT_STRIDE,
                    lookback=LW,horizons=H) for s in segs]
tw=ros(lambda: AnalyticFOPDTPlant(P)); bo=ros(lambda: AnalyticFOPDTPlant(BO)); idl=ros(lambda: IdealKinematicPlant())
k=H.index(24)
def perev(A,B):
    T_=build_error_table(A,B,params=P,mcfg=mcfg); m=(np.array(T_.bin)=='trans')
    sid=np.array(T_.seg_id)[m]; a=T_.epos_twin[m,k]; b=T_.epos_def[m,k]
    return (np.array([np.median(a[sid==s]) for s in sorted(set(sid))]),
            np.array([np.median(b[sid==s]) for s in sorted(set(sid))]))
tw_e,bo_e = perev(tw,bo)
_,   id_e = perev(tw,idl)
def rep(x,y,la,lb):
    d=y-x; st,p=wilcoxon(x,y,alternative='less'); pos=int((d>0).sum()); neg=int((d<0).sum())
    return dict(a=la,b=lb,med_a=float(np.median(x)),med_b=float(np.median(y)),
                gap=float(np.median(d)),p=float(p),wins=pos,losses=neg,rb=(pos-neg)/len(d))
r1=rep(tw_e,bo_e,'per-DOF','bayesopt'); r2=rep(bo_e,id_e,'bayesopt','ideal'); r3=rep(tw_e,id_e,'per-DOF','ideal')
ts=datetime.now().strftime('%Y%m%d_%H%M')
L=[f"# Controle negativo (bayesopt) com estatistica POR EVENTO - {ts}","",
 "Mesmo padrao do resultado principal: mediana por evento, n=10, h_ref=0.4 s,",
 "bin translacional. Substitui os numeros por janela que nao eram comparaveis.","",
 "| comparacao | mediana A | mediana B | gap mediano | vitorias A | Wilcoxon p | rank-biserial |",
 "|---|---|---|---|---|---|---|"]
for r in (r3,r2,r1):
    L.append(f"| {r['a']} vs {r['b']} | {r['med_a']:.4f} | {r['med_b']:.4f} | {r['gap']:+.4f} | "
             f"{r['wins']}/10 | {r['p']:.4g} | {r['rb']:+.3f} |")
L+=["","## Leitura","",
 f"- per-DOF vs ideal: gap {r3['gap']:+.4f} m, {r3['wins']}/10, p={r3['p']:.3g} -> resultado principal, robusto.",
 f"- bayesopt vs ideal: gap {r2['gap']:+.4f} m, {r2['wins']}/10, p={r2['p']:.3g}.",
 f"- **per-DOF vs bayesopt: gap {r1['gap']:+.4f} m, {r1['wins']}/10 eventos, p={r1['p']:.3g}, "
 f"efeito {r1['rb']:+.3f}**.","",
 "Se o gate do artigo (efeito >= 0.2 E p < 0.05) for aplicado a per-DOF vs bayesopt,",
 "o veredito e o que aparece na ultima linha -- e e ELE que deve ser reportado, nao",
 "o p por janela de 0.040 usado antes (objecao C3 da revisao V2)."]
o=R/'results'/f'negcontrol_perevent_{ts}.md'; o.write_text("\n".join(L),encoding='utf-8')
print("\n".join(L))
