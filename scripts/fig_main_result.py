import sys,pickle
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
from phase0_replay.splits import load_holdout_segments
from phase0_replay.params import load_twin_params
from phase0_replay.metrics import load_metrics_config, build_error_table
from phase0_replay.plant import AnalyticFOPDTPlant, IdealKinematicPlant
from phase0_replay.rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
LW=15; P=load_twin_params(); mcfg=load_metrics_config(); H=list(mcfg['horizons'])
segs,_=load_holdout_segments(min_displacement_m=0.1)
rs=pickle.load((R/'results/_cache_rsim_rollouts.pkl').open('rb'))
def ros(f): return [rollout_segment(s,f(),h_max=DEFAULT_HMAX,stride=DEFAULT_STRIDE,
                    lookback=LW,horizons=H) for s in segs]
tw=ros(lambda: AnalyticFOPDTPlant(P)); idl=ros(lambda: IdealKinematicPlant())
rsl=[rs[str(i)] for i in range(len(segs))]
T1=build_error_table(tw,idl,params=P,mcfg=mcfg)
T2=build_error_table(tw,rsl,params=P,mcfg=mcfg)
m=(np.array(T1.bin)=='trans'); hs=np.array(H)/60.0
tw_m=[np.median(T1.epos_twin[m,k]) for k in range(len(H))]
id_m=[np.median(T1.epos_def[m,k]) for k in range(len(H))]
rs_m=[np.median(T2.epos_def[m,k]) for k in range(len(H))]
sid=np.array(T1.seg_id)[m]; k=H.index(24)
et=T1.epos_twin[m,k]; ed=T1.epos_def[m,k]
ev=[(np.median(et[sid==s]),np.median(ed[sid==s])) for s in sorted(set(sid))]
fig,ax=plt.subplots(1,2,figsize=(7.0,2.5),gridspec_kw={'width_ratios':[1.25,1]})
a=ax[0]
a.plot(hs,id_m,'o-',color='#1f77b4',lw=1.4,ms=3.5,label='ideal kinematics')
a.plot(hs,rs_m,'s:',color='#ff7f0e',lw=1.4,ms=3.5,label='rSim (untuned ODE)')
a.plot(hs,tw_m,'^--',color='#d62728',lw=1.6,ms=4,label='FOPDT model')
a.axvline(0.4,color='0.6',ls=':',lw=0.9)
a.text(0.42,0.012,r'$h_{ref}$',fontsize=7,color='0.4')
a.set_xlabel('horizon (s)',fontsize=8); a.set_ylabel(r'median $e_{pos}$ (m)',fontsize=8)
a.tick_params(labelsize=7); a.grid(alpha=.3,lw=.4); a.legend(fontsize=7,loc='upper left')
a.set_title('(a) free-run position error',fontsize=8)
b=ax[1]
for i,(t,d) in enumerate(ev):
    b.plot([0,1],[d,t],'-',color='0.75',lw=0.8,zorder=1)
b.scatter([0]*len(ev),[d for _,d in ev],s=22,color='#1f77b4',zorder=3,label='ideal')
b.scatter([1]*len(ev),[t for t,_ in ev],s=22,color='#d62728',marker='^',zorder=3,label='model')
b.set_xlim(-0.35,1.35); b.set_xticks([0,1]); b.set_xticklabels(['ideal','model'],fontsize=8)
b.set_ylabel(r'per-event median $e_{pos}$ (m)',fontsize=8)
b.tick_params(labelsize=7); b.grid(alpha=.3,lw=.4,axis='y')
b.set_title(r'(b) 10 hold-out events, $h_{ref}$',fontsize=8)
plt.tight_layout(pad=0.4)
o=R/'results/figs/main_result_en.png'; plt.savefig(o,dpi=300,bbox_inches='tight')
import shutil; shutil.copy(o,R/'latex/figures/main_result_en.png')
print('gerada. twin',[f'{v:.4f}' for v in tw_m]); print('ideal',[f'{v:.4f}' for v in id_m]); print('rsim',[f'{v:.4f}' for v in rs_m])
