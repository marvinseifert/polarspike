import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys
# usage: python plot_rf_vs_bar_axis.py rf_vs_bar_axis.csv output.png
CSV, OUT = sys.argv[1], sys.argv[2]
o=pd.read_csv(CSV); o['c']=np.cos(np.deg2rad(2*o['diff']))
COL={'12px':'#2a78d6','32px':'#eb6834'}; INK='#2b2b2a'; MUTED='#6b6a64'; GRID='#e4e2dc'
plt.rcParams.update({'font.size':10,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
                     'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
def ci(x):
    x=np.asarray(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan; return m,1.96*se
fig,axs=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw=dict(width_ratios=[1.1,1.3,1.1],wspace=0.32))
# A: distribution of angle between RF axis and preferred motion axis
ax=axs[0]; bins=np.arange(0,91,15)
for px in ('12px','32px'):
    t=o[(o.elong>1.2)&(o.px==px)]['diff']; h,_=np.histogram(t,bins); frac=h/h.sum()
    ax.stairs(frac,bins,color=COL[px],lw=2,label=f'{px} noise (n = {len(t)})')
ax.axhline(1/6,color=MUTED,ls='--',lw=1); ax.text(89,1/6+0.006,'no relation',ha='right',va='bottom',color=MUTED,fontsize=9)
ax.set_xticks(bins); ax.set_xlim(0,90); ax.set_xlabel('Angle between RF long axis and preferred\nbar motion axis (°)')
ax.set_ylabel('Fraction of cells'); ax.set_title('A  Cells with elongation > 1.2',loc='left',color=INK,fontsize=11)
ax.set_ylim(0,0.36); ax.text(1,0.355,'← parallel',color=MUTED,fontsize=9,va='top'); ax.text(89,0.355,'perpendicular →',color=MUTED,fontsize=9,va='top',ha='right')
ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0.0,0.92))
# B: per recording
ax=axs[1]
def dkey(r): d,m,y=map(int,r.split('_')); return (y,m,d)
recs=sorted(o.rec.unique(),key=dkey)
for i,r in enumerate(recs):
    t=o[(o.rec==r)&(o.elong>1.2)]; m,e=ci(t.c); px=t.px.iloc[0]
    ax.errorbar(i,m,yerr=e,fmt='o',ms=7,color=COL[px],ecolor=COL[px],elinewidth=1.5,capsize=0)
    ax.text(i,-1.04,f'n={len(t)}',ha='center',va='bottom',color=MUTED,fontsize=8)
ax.axhline(0,color=MUTED,lw=1,ls='--'); ax.set_ylim(-1.08,1); ax.set_xlim(-0.6,len(recs)-0.4)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=COL[k],label=f'{k} noise') for k in COL],frameon=False,loc='upper left')
ax.set_xticks(range(len(recs))); ax.set_xticklabels([r[:5].replace('_','/')+'/'+r[8:] for r in recs],rotation=40,ha='right')
ax.set_ylabel('Alignment score (mean cos 2Δ)\n+1 parallel · 0 none · −1 perpendicular')
ax.set_title('B  Each recording (elongation > 1.2, mean ± 95% CI)',loc='left',color=INK,fontsize=11)
ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True)
i15=recs.index('15_10_2026'); ax.annotate('after 8 Oct 2026\nprojection change',(i15,ci(o[(o.rec=='15_10_2026')&(o.elong>1.2)].c)[0]),xytext=(i15-1.6,0.55),
    color=MUTED,fontsize=8,arrowprops=dict(arrowstyle='-',color=MUTED,lw=0.8))
# C: score vs elongation threshold
ax=axs[2]; th=np.round(np.arange(1.0,1.61,0.05),2)
for px in ('12px','32px'):
    ms,es,ns=[],[],[]
    for e in th:
        t=o[(o.elong>e)&(o.px==px)].c; m,er=ci(t) if len(t)>2 else (np.nan,np.nan); ms.append(m); es.append(er); ns.append(len(t))
    ms,es=np.array(ms),np.array(es)
    ax.fill_between(th,ms-es,ms+es,color=COL[px],alpha=0.15,lw=0); ax.plot(th,ms,color=COL[px],lw=2)
    ax.text(th[-1]+0.015,{'12px':0.05,'32px':-0.32}[px],f'{px}\nn = {ns[0]}→{ns[-1]}',color=COL[px],fontsize=9,va='center')
ax.axhline(0,color=MUTED,lw=1,ls='--'); ax.set_xlim(1.0,1.72); ax.set_ylim(-1,1)
ax.set_xlabel('Only cells with elongation above'); ax.set_ylabel('Alignment score (mean cos 2Δ)')
ax.set_title('C  Depends on how elongated',loc='left',color=INK,fontsize=11)
ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True)
fig.suptitle('RF long axis vs preferred moving-bar axis (door view): RFs tend to lie perpendicular to the preferred motion, i.e. along the bar',
             x=0.06,ha='left',color=INK,fontsize=12,y=1.02)
fig.savefig(OUT,dpi=150,bbox_inches='tight',facecolor='white')
print('ok')
