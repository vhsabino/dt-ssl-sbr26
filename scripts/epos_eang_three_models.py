import sys,pickle,json
from pathlib import Path
from datetime import datetime
import numpy as np
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
def tab(a,b): return build_error_table(a,b,params=P,mcfg=mcfg)
T1=tab(tw,idl); T2=tab(tw,rsl)
m=(np.array(T1.bin)=='trans')
ts=datetime.now().strftime('%Y%m%d_%H%M')
L=[f"# e_ang por horizonte e e_pos dos tres modelos - {ts}","",
 "Bin translacional, medianas sobre janelas densas.","",
 "| h (s) | twin e_pos | ideal e_pos | rSim e_pos | twin e_ang | ideal e_ang | rSim e_ang |",
 "|---|---|---|---|---|---|---|"]
for k,h in enumerate(H):
    L.append(f"| {h/60:.2f} | {np.median(T1.epos_twin[m,k]):.4f} | {np.median(T1.epos_def[m,k]):.4f} "
             f"| {np.median(T2.epos_def[m,k]):.4f} | {np.median(T1.eang_twin[m,k]):.4f} "
             f"| {np.median(T1.eang_def[m,k]):.4f} | {np.median(T2.eang_def[m,k]):.4f} |")
L+=["","e_ang em rad. Fecha a objecao C6 (o artigo afirmava que o heading do twin",
 "excede o do rSim em horizontes longos sem nunca dar o numero do rSim)."]
o=R/'results'/f'epos_eang_three_models_{ts}.md'; o.write_text("\n".join(L),encoding='utf-8')
print("\n".join(L))
# --- checagem de vazamento ---
ho=json.load(open(R/'config/holdout_events.json'))['events']
fid={e['event'] for e in ho}
import csv
idh=set()
with open(R/'results/fit_holdout_vxvy_20260731_024049.csv') as f:
    for row in csv.DictReader(f): idh.add(row['evento'])
print("\n=== VAZAMENTO ===")
print("hold-out de fidelidade (10):", sorted(fid))
print("hold-out de identificacao   :", sorted(idh))
print("ID subset de FID?", idh <= fid)
print("eventos de fidelidade NAO usados na ID:", sorted(fid-idh))
