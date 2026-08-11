import sys,pickle
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
from phase0_replay.splits import load_holdout_segments
from phase0_replay.params import load_twin_params
from phase0_replay.metrics import load_metrics_config
from phase0_replay.plant import AnalyticFOPDTPlant, IdealKinematicPlant
from phase0_replay.rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
LW=15; p=load_twin_params(); mcfg=load_metrics_config(); H=list(mcfg['horizons'])
segs,_=load_holdout_segments(min_displacement_m=0.1)
def ros(f): return [rollout_segment(s,f(),h_max=DEFAULT_HMAX,stride=DEFAULT_STRIDE,
                    lookback=LW,horizons=H) for s in segs]
tw=ros(lambda: AnalyticFOPDTPlant(p)); idl=ros(lambda: IdealKinematicPlant())
rs=pickle.load((R/'results/_cache_rsim_rollouts.pkl').open('rb'))
# escolhe a janela cujo erro do gemeo em h=90 e a MEDIANA (representativa, nao cherry-pick)
cand=[]
for si in range(len(segs)):
    for wi,w in enumerate(tw[si].windows):
        e=np.hypot(*(w.pose_sim[90][:2]-w.pose_real[90][:2])); cand.append((e,si,wi))
cand.sort(); e,si,wi=cand[len(cand)//2]
print(f'janela mediana: seg={si} win={wi} e_pos(1.5s)={e:.4f} m')
wa=tw[si].windows[wi]; wb=idl[si].windows[wi]; wc=rs[str(si)].windows[wi]
real=wa.pose_real; t=np.arange(real.shape[0])/60.0
fig,ax=plt.subplots(figsize=(2.5,3.0))
def anch(P): return P[:,0]-P[0,0], P[:,1]-P[0,1]
for P,st,lab,lw_ in [(real,'k-','measured',1.8),(wa.pose_sim,'r--','twin (FOPDT)',1.5),
                     (wb.pose_sim,'-.','ideal kinematics',1.2),(wc.pose_sim,':','rSim (ODE)',1.4)]:
    x,y=anch(P); ax.plot(x,y,st,lw=lw_,label=lab)
x,y=anch(real); ax.plot(x[0],y[0],'ko',ms=4)
ax.set_xlabel('x displacement from anchor (m)',fontsize=8)
ax.set_ylabel('y displacement (m)',fontsize=8)
ax.tick_params(labelsize=7); ax.grid(alpha=.3,lw=.4); ax.axis('equal')
ax.legend(fontsize=6.5,loc='upper left',framealpha=.9)
plt.tight_layout(pad=0.3)
o=R/'results/figs/trajectory_holdout_en.png'; plt.savefig(o,dpi=300,bbox_inches='tight')
import shutil; shutil.copy(o,R/'latex/figures/trajectory_holdout_en.png')
print('gerada')
