import sys, math, itertools, json
from pathlib import Path
from datetime import datetime
import numpy as np
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
from phase0_replay.splits import load_holdout_segments
from phase0_replay.params import load_twin_params
from phase0_replay.metrics import load_metrics_config, build_error_table
from phase0_replay.plant import AnalyticFOPDTPlant, IdealKinematicPlant, Plant
from phase0_replay.rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
LW=15; P=load_twin_params(); mcfg=load_metrics_config(); H=list(mcfg['horizons'])
segs,_=load_holdout_segments(min_displacement_m=0.1)
HREF=H.index(24)
class DP:
    def __init__(s,K,tau,Td): s.K,s.tau,s.Td=K,tau,Td; s.placeholder=False
def variant(uK,ut,uD):
    return {d:DP(P[d].K if uK else 1.0, P[d].tau if ut else 0.0, P[d].Td if uD else 0.0)
            for d in ('vx','vy','w')}
class ColdFOPDT(AnalyticFOPDTPlant):
    """Identico ao gemeo, mas SEM a condicao inicial de velocidade medida:
    parte do repouso, como o rSim."""
    def reset(self, pose0, velb0=(0.,0.,0.)): super().reset(pose0,(0.,0.,0.))
class ConstVel(Plant):
    """Extrapolacao de velocidade constante: usa SO a IC medida, ignora comandos.
    E o comparador honesto para separar IC de dinamica identificada."""
    def reset(self, pose0, velb0=(0.,0.,0.)):
        self.x,self.y,self.th=(float(v) for v in pose0)
        self._v=np.asarray(velb0,float).reshape(3); self._vbody=self._v.copy()
    def prime_delay(self, prev_cmds): pass
    def step(self, cmd, dt):
        th=self.th+dt*self._v[2]; c,s=math.cos(th),math.sin(th)
        self.x+=dt*(c*self._v[0]-s*self._v[1]); self.y+=dt*(s*self._v[0]+c*self._v[1])
        self.th=th; return (self.x,self.y,self.th)
def med(factory):
    ro=[rollout_segment(sg,factory(),h_max=DEFAULT_HMAX,stride=DEFAULT_STRIDE,
        lookback=LW,horizons=H) for sg in segs]
    T=build_error_table(ro,ro,params=P,mcfg=mcfg)
    m=(np.array(T.bin)=='trans')
    return float(np.median(T.epos_twin[m,HREF])), T, m
# ---- fatorial completo 2^3 (IC medida) ----
E={}
for uK,ut,uD in itertools.product([0,1],repeat=3):
    v=variant(uK,ut,uD); E[(uK,ut,uD)]=med(lambda v=v: AnalyticFOPDTPlant(v))[0]
def shapley(idx):
    tot=0.0; others=[i for i in range(3) if i!=idx]
    for bits in itertools.product([0,1],repeat=2):
        s=[0,0,0]
        for k,i in enumerate(others): s[i]=bits[k]
        a=list(s); a[idx]=0; b=list(s); b[idx]=1
        w=(math.factorial(sum(bits))*math.factorial(2-sum(bits)))/math.factorial(3)
        tot+=w*(E[tuple(a)]-E[tuple(b)])     # reducao de erro
    return tot
names=['gain K','time constant tau','dead time Td']
sh=[shapley(i) for i in range(3)]
# ---- bracos de condicao inicial ----
full=variant(1,1,1)
e_full=E[(1,1,1)]; e_ideal=E[(0,0,0)]
e_cold=med(lambda: ColdFOPDT(full))[0]
e_cv  =med(lambda: ConstVel())[0]
ts=datetime.now().strftime('%Y%m%d_%H%M')
L=[f"# Ablacao ordem-robusta + condicao inicial - {ts}","",
 f"Mediana de e_pos (m) no bin translacional, h_ref = 0.4 s, mesmas janelas.","",
 "## 1. Fatorial completo 2^3 (todas as 8 combinacoes, IC medida)","",
 "| K | tau | Td | e_pos (m) |","|---|---|---|---|"]
for k in sorted(E): L.append(f"| {'on' if k[0] else 'off'} | {'on' if k[1] else 'off'} | {'on' if k[2] else 'off'} | {E[k]:.4f} |")
L+=["","## 2. Atribuicao de Shapley (ordem-INDEPENDENTE)","",
 "Reducao media de e_pos atribuivel a cada grupo, sobre todas as ordens:","",
 "| grupo | contribuicao (m) | fracao da melhoria |","|---|---|---|"]
tot=sum(sh)
for n,s in zip(names,sh): L.append(f"| {n} | {s:+.4f} | {100*s/tot:+.0f}% |")
L+=["",f"Soma = {tot:+.4f} m = melhoria total (ideal {e_ideal:.4f} -> completo {e_full:.4f}).",
 "","## 3. Quanto da vitoria e condicao inicial, e nao dinamica","",
 "| planta | IC | e_pos (m) |","|---|---|---|",
 f"| ideal kinematics (memoryless) | irrelevante | {e_ideal:.4f} |",
 f"| **const-velocity extrapolation** | **usa SO a IC medida** | **{e_cv:.4f}** |",
 f"| FOPDT completo | parte do repouso (como o rSim) | {e_cold:.4f} |",
 f"| FOPDT completo | IC medida | {e_full:.4f} |","",
 f"- A extrapolacao de velocidade constante, que usa a IC e IGNORA os comandos, da {e_cv:.4f} m.",
 f"- O gemeo sem IC medida da {e_cold:.4f} m; com IC medida, {e_full:.4f} m.",
 f"- Logo a IC sozinha explica {e_cold-e_full:+.4f} m e a dinamica identificada o restante."]
o=R/'results'/f'ablation_orderfree_{ts}.md'; o.write_text("\n".join(L),encoding='utf-8')
print("\n".join(L)); print("\nartefato:",o.name)
