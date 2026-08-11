import json,math,sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
R=Path('/sessions/happy-modest-euler/mnt/digitaltwin')
p=json.load(open(R/'config/twin_params.json'))
K,tau,Td=p['dof']['w']['K'],p['dof']['w']['tau'],p['dof']['w']['Td']
dt=p['ts']; S=R/'data/extracted/2026-05-18_19-2-15/splits/rotate'
# 3 painéis: 1 treino + 2 hold-out (split de results/identificacao_omega_20260724.md)
PANELS=[('rotate_01_09','train','+5.0'),('rotate_01_06','hold-out','-1.0'),('rotate_01_07','hold-out','+2.5')]
def sim(u,t):
    m=math.ceil(Td/dt); y=np.zeros(len(u)); a=1-math.exp(-dt/tau); v=0.0
    for k in range(len(u)):
        ud=u[k-m] if k-m>=0 else 0.0
        v=v+a*(ud-v); y[k]=K*v
    return y
fig,axes=plt.subplots(3,1,figsize=(3.5,4.6),sharex=True)
for ax,(ev,kind,amp) in zip(axes,PANELS):
    c=pd.read_csv(S/ev/'commands.csv'); r=pd.read_csv(S/ev/'processed_robots.csv')
    tc=c['timestamp'].to_numpy(); u=c['move_w'].to_numpy()
    tr=r['timestamp'].to_numpy(); th=np.unwrap(r['position_w'].to_numpy())
    t0=tc[0]; tg=np.arange(0,min(tc[-1],tr[-1])-t0,dt)
    ug=np.interp(tg,tc-t0,u); thg=np.interp(tg,tr-t0,th)
    w=7 if len(thg)>7 else (len(thg)//2*2-1)
    meas=np.gradient(savgol_filter(thg,w,2))/dt
    ax.plot(tg,meas,color='0.25',lw=0.7,label='measured')
    ax.plot(tg,sim(ug,tg),'r--',lw=1.3,label='FOPDT model')
    ax.set_ylabel(r'$\omega$ (rad/s)',fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=.3,lw=.4)
    ax.set_title(f'{amp} rad/s step ({kind})',fontsize=8,pad=2)
axes[0].legend(fontsize=7,loc='lower right',framealpha=.9)
axes[-1].set_xlabel('time from step onset (s)',fontsize=8)
plt.tight_layout(pad=0.3)
out=R/'results/figs/omega_stepresponse_en.png'
plt.savefig(out,dpi=300,bbox_inches='tight')
import shutil; shutil.copy(out,R/'latex/figures/omega_stepresponse_en.png')
print('gerada:',out.name)
