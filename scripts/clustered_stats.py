import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import wilcoxon
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin'); sys.path.insert(0,str(R))
from phase0_replay.splits import load_holdout_segments
from phase0_replay.params import load_twin_params
from phase0_replay.metrics import load_metrics_config, build_error_table
from phase0_replay.plant import AnalyticFOPDTPlant, IdealKinematicPlant
from phase0_replay.rollout import rollout_segment, DEFAULT_HMAX, DEFAULT_STRIDE
LW=15; p=load_twin_params(); mcfg=load_metrics_config(); H=list(mcfg['horizons'])
segs,_=load_holdout_segments(min_displacement_m=0.1)
def ros(f): return [rollout_segment(s,f(),h_max=DEFAULT_HMAX,stride=DEFAULT_STRIDE,
                    lookback=LW,horizons=H) for s in segs]
T=build_error_table(ros(lambda: AnalyticFOPDTPlant(p)), ros(lambda: IdealKinematicPlant()),
                    params=p, mcfg=mcfg)
hs=list(np.asarray(mcfg['horizons'],int)); k=hs.index(24)
m=(np.array(T.bin)=='trans')
sid=np.array(T.seg_id)[m]; et=T.epos_twin[m,k]; ed=T.epos_def[m,k]
indep=np.array(T.indep)[m]
ts=T.ts; hmax=DEFAULT_HMAX
rows=[]
for s in sorted(set(sid)):
    q=sid==s
    rows.append((int(s), float(np.median(et[q])), float(np.median(ed[q])), int(q.sum())))
tw=np.array([r[1] for r in rows]); idl=np.array([r[2] for r in rows]); d=idl-tw
st,pv=wilcoxon(tw,idl,alternative='less')
rng=np.random.default_rng(42)
bs=np.array([np.median(d[rng.integers(0,len(d),len(d))]) for _ in range(10000)])
ci=(np.percentile(bs,2.5),np.percentile(bs,97.5))
pos=int((d>0).sum()); neg=int((d<0).sum()); rb=(pos-neg)/len(d)
n_indep=int(indep.sum()); secs=n_indep*hmax*ts
stamp=datetime.now().strftime('%Y%m%d_%H%M')
L=[f"# Estatistica agrupada por EVENTO — {stamp}","",
 f"Unidade de analise = evento de hold-out. n = **{len(rows)} eventos** "
 f"(contra n=49 janelas na analise por janela).","",
 f"Janelas translacionais densas: {int(m.sum())}; nao-sobrepostas: {n_indep}; "
 f"movimento pontuado = {n_indep} x {hmax*ts:.1f} s = **{secs:.1f} s**.","",
 "| evento (idx) | twin e_pos (m) | ideal e_pos (m) | gap (m) | janelas |","|---|---|---|---|---|"]
for s,t,i,n in rows: L.append(f"| {s} | {t:.4f} | {i:.4f} | {i-t:+.4f} | {n} |")
L+=["",f"- Wilcoxon pareado **por evento** (one-sided twin<ideal): **p = {pv:.4g}**",
 f"- Gap mediano por evento: **{np.median(d):+.4f} m**; IC95% bootstrap por evento "
 f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
 f"- Rank-biserial pareado por evento: **{rb:+.3f}** ({pos} a favor / {neg} contra)",
 "", "Todos em h_ref = 0.4 s, bin translacional, twin vs cinematica ideal."]
o=R/'results'/f'clustered_stats_{stamp}.md'; o.write_text("\n".join(L),encoding='utf-8')
print("\n".join(L))
