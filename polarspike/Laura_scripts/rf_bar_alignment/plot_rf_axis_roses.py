import numpy as np, pandas as pd, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys
# usage: python plot_rf_axis_roses.py rf_vs_bar_axis.csv output.png
CSV, OUT = sys.argv[1], sys.argv[2]
o=pd.read_csv(CSV)
sel=json.load(open('/home/laura/PycharmProjects/polarspike/polarspike/Laura_scripts/master_stimulus_selection.json'))
reg={r:(c.get('region') or sel.get('regions',{}).get(f"{r}/{c.get('phase')}") or 'unknown') for r,c in sel['recordings'].items()}
COL={'12px':'#2a78d6','32px':'#eb6834'}; INK='#2b2b2a'; MUTED='#6b6a64'
E=1.2; o=o[o.elong>E]
def ray(th):
    z=np.exp(2j*np.deg2rad(th)); n=len(th); R=abs(z.mean()); Z=n*R*R
    p=np.exp(-Z)*(1+(2*Z-Z*Z)/(4*n)); return R,min(max(p,0),1),(np.degrees(np.angle(z.mean()))/2)%180
def dkey(r): d,m,y=map(int,r.split('_')); return (y,m,d)
recs=sorted(o.rec.unique(),key=dkey)
panels=[('All 12 px','px','12px'),('All 32 px','px','32px')]+[(r,'rec',r) for r in recs]
fig,axs=plt.subplots(2,6,figsize=(17,7.6),subplot_kw=dict(projection='polar'))
axs=axs.ravel()
bw=15; edges=np.deg2rad(np.arange(0,361,bw))
for ax,(title,col,val) in zip(axs,panels):
    t=o[o[col]==val]; th=t.rf_axis.values; px=t.px.iloc[0]
    both=np.concatenate([th,th+180])  # an axis points both ways
    h,_=np.histogram(np.deg2rad(both)%(2*np.pi),edges)
    ax.bar(edges[:-1]+np.deg2rad(bw/2),h/len(th),width=np.deg2rad(bw)*0.92,color=COL[px],alpha=0.85,lw=0)
    R,p,m=ray(th)
    if p<0.05:
        for a in (m,m+180): ax.plot([0,np.deg2rad(a)],[0,(h/len(th)).max()*1.05],color=INK,lw=1.6)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0,45,90,135,180,225,270,315])); ax.set_xticklabels(['up','','right','','down','','left',''],color=MUTED,fontsize=8)
    ax.set_yticklabels([]); ax.grid(color='#e4e2dc',lw=0.6); ax.spines['polar'].set_color('#d6d4cd')
    name=title if col=='px' else title.replace('_','/')
    sub=(f"{reg.get(val,'')} · " if col=='rec' else '')+f"n={len(th)}"
    ptxt='p<0.001' if p<0.001 else f'p={p:.1g}'
    sig=f"axis {m:.0f}°, strength {R:.2f}, {ptxt}" if p<0.05 else f"no clear axis ({ptxt})"
    ax.set_title(f"{name}\n{sub}\n{sig}",fontsize=9,color=INK,pad=14)
for ax in axs[len(panels):]: ax.set_visible(False)
fig.suptitle(f'RF long-axis directions (door view; cells with elongation > {E}). Black line = mean axis where significant (p < 0.05).',
             x=0.02,ha='left',color=INK,fontsize=12,y=1.01)
fig.text(0.02,-0.02,'Bars are mirrored (an axis has no front or back). Blue = 12 px noise, orange = 32 px. 0° = up–down, 90° = left–right.',color=MUTED,fontsize=9)
fig.tight_layout(h_pad=3.5); fig.savefig(OUT,dpi=140,bbox_inches='tight',facecolor='white'); print('ok')
