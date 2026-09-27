"""One fixed allocation maximizing posterior expected revenue; no model refit."""
from pathlib import Path
import hashlib
import argparse
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prior_predictive import ROOT, LAG, prepare, adstock

parser=argparse.ArgumentParser()
parser.add_argument('--posterior-report',default='mcmc')
parser.add_argument('--channel-report',default='channel_results')
parser.add_argument('--output-report',default='budget_optimization')
parser.add_argument('--calibrated',action='store_true')
options=parser.parse_args()
OUT=ROOT/'reports'/options.output_report
OUT.mkdir(parents=True,exist_ok=True)
p=prepare(); n=p['cutoff']; sl=slice(LAG,n)
table=pd.read_csv(ROOT/'reports'/options.channel_report/'channel_results.csv')
spend=table.spend.to_numpy(); budget=spend.sum(); weights=spend/budget
raw=pd.read_csv(ROOT/'data/geo_all_channels.csv').set_index(['geo','time']).reindex(pd.MultiIndex.from_product([p['geos'],p['dates']]))
rpc=raw.revenue_per_conversion.to_numpy().reshape(p['y'].shape)[:,sl]
revenue_mult=p['pop'][:,sl]*p['yscale']*rpc
conversion_mult=p['pop'][:,sl]*p['yscale']
path=ROOT/'reports'/options.posterior_report/'posterior_samples.npz'
with np.load(path) as f:
    samples={k:f[k] for k in ['decay','k','beta']}
chains,draws=samples['decay'].shape[:2]
x=p['x'][:,:n].copy(); xp=x.copy(); xp[:,LAG:]=0

def components(c,d):
    k=samples['k'][c,d]
    pre=adstock(xp,samples['decay'][c,d])[:,sl]
    window=adstock(x,samples['decay'][c,d])[:,sl]-pre
    return window/(pre+k),samples['beta'][c,d,:,None,:]*k/(pre+k)

def evaluate(v,w,s,mult):
    vs=v*s
    revenue=(w*vs/(1+vs)*mult[:,:,None]).sum(axis=(0,1))
    deriv=(w*v/(1+vs)**2*mult[:,:,None]).sum(axis=(0,1))
    return revenue,deriv

# Exact draw-level response and derivative evaluations on a bounded scale grid.
grid=np.linspace(.8,1.2,41)
curves=np.empty((chains,draws,len(grid),5)); derivs=np.empty_like(curves)
for c in range(chains):
    for d in range(draws):
        v,w=components(c,d)
        for i,s in enumerate(grid):
            curves[c,d,i],derivs[c,d,i]=evaluate(v,w,s,revenue_mult)
    print(f'Response curves: chain {c+1}/{chains}',flush=True)
mean=curves.mean(axis=(0,1)); grad=derivs.mean(axis=(0,1))
spline=CubicHermiteSpline(grid,mean,grad,axis=0)
idx=np.arange(5)
def objective(s):return -float(spline(s)[idx,idx].sum()/1e6)
def jac(s):return -spline(s,1)[idx,idx]/1e6
constraint={'type':'eq','fun':lambda s: weights@s-1,'jac':lambda s:weights}
result=minimize(objective,np.ones(5),jac=jac,bounds=[(.8,1.2)]*5,constraints=[constraint],method='SLSQP',options={'ftol':1e-12,'maxiter':300})
assert result.success,result.message
allocation=result.x
assert abs(weights@allocation-1)<1e-9 and np.all((allocation>=.8-1e-9)&(allocation<=1.2+1e-9))
# Evaluate the SAME allocation exactly in every draw, paired with the original.
base=np.empty((chains,draws,5)); opt=np.empty_like(base); mroas=np.empty_like(base); conv_gain=np.empty_like(base)
for c in range(chains):
    for d in range(draws):
        v,w=components(c,d)
        base[c,d],_=evaluate(v,w,np.ones(5),revenue_mult)
        opt[c,d],der=evaluate(v,w,allocation,revenue_mult)
        mroas[c,d]=der/spend
        conv_gain[c,d]=evaluate(v,w,allocation,conversion_mult)[0]-evaluate(v,w,np.ones(5),conversion_mult)[0]
with np.load(ROOT/'reports'/options.channel_report/'posterior_metrics.npz') as prior:
    np.testing.assert_allclose(base,prior['incremental_revenue'],rtol=1e-10,atol=1e-6)
exact_marginal=mroas.mean(axis=(0,1))
np.testing.assert_allclose(-jac(allocation)*1e6/spend,exact_marginal,rtol=1e-6,atol=1e-7)
# Concavity makes feasible KKT conditions sufficient for the global optimum.
free=(allocation>.800001)&(allocation<1.199999)
assert free.any()
shadow=float(exact_marginal[free].mean())
assert np.max(np.abs(exact_marginal[free]-shadow))<1e-5
assert np.all(exact_marginal[allocation<=.800001]<=shadow+1e-5)
assert np.all(exact_marginal[allocation>=1.199999]>=shadow-1e-5)
gain=(opt-base).sum(axis=2)
assert gain.mean()>=0
np.savez_compressed(OUT/'posterior_comparison.npz',baseline_revenue=base,optimized_revenue=opt,revenue_gain=gain,conversion_gain=conv_gain.sum(axis=2),optimized_marginal_roas=mroas,allocation_scale=allocation)
np.savez_compressed(OUT/'response_curves.npz',scale=grid,mean=mean,lo90=np.quantile(curves,.05,axis=(0,1)),hi90=np.quantile(curves,.95,axis=(0,1)))
def summary(a):return dict(mean=float(a.mean()),lo90=float(np.quantile(a,.05)),hi90=float(np.quantile(a,.95)))
rows=[]
for j in range(5):
    rows.append(dict(channel=f'Channel{j}',original_spend=spend[j],optimized_spend=spend[j]*allocation[j],change_pct=100*(allocation[j]-1),original_marginal_roas=float(grad[20,j]/spend[j]),optimized_marginal_roas=exact_marginal[j],bound='lower' if allocation[j]<.800001 else ('upper' if allocation[j]>1.199999 else 'interior')))
pd.DataFrame(rows).to_csv(OUT/'allocation.csv',index=False)
diag=az.summary(az.from_dict(posterior={'revenue_gain':gain}),kind='diagnostics',round_to='none')
diag.to_csv(OUT/'gain_diagnostics.csv')
meta=dict(model=('Hierarchical MMM calibrated with hypothetical experimental evidence' if options.calibrated else 'Frozen uncalibrated hierarchical MMM'),window=[str(p['dates'][LAG]),str(p['dates'][n-1])],budget=float(budget),bounds=[.8,1.2],objective='Maximize posterior expected in-window incremental revenue; one common allocation across all draws',assumptions=['Historical proportional geo-week flighting within each channel','Fixed cell-level impression prices and revenue per conversion','Earlier carryover and controls fixed','In-window outcomes only; excludes carryout beyond window',('Hypothetical experiment calibration; simulated data; no refit during optimization' if options.calibrated else 'No experiment calibration; no model refit; simulated data')],baseline_incremental_revenue=summary(base.sum(axis=2)),optimized_incremental_revenue=summary(opt.sum(axis=2)),revenue_gain=summary(gain),conversion_gain=summary(conv_gain.sum(axis=2)),gain_as_pct_baseline_paid_revenue=summary(100*gain/base.sum(axis=2)),probability_revenue_gain_positive=float((gain>0).mean()),probability_positive_mcse=float(az.mcse((gain>0).astype(float)).item()),shadow_marginal_roas=shadow,optimizer_message=result.message,max_rhat=float(diag.r_hat.max()),min_ess_bulk=float(diag.ess_bulk.min()),checks=['Fixed total budget and all bounds satisfied','Baseline matches channel table draw by draw','Exact gradient agrees with interpolated optimizer gradient','Exact-gradient KKT conditions verified for concave objective','Common allocation evaluated on paired posterior draws'],sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [path,ROOT/'data/geo_all_channels.csv',Path(__file__)]})
(OUT/'summary.json').write_text(json.dumps(meta,indent=2))
lines=['| Channel | Original spend (million CU) | Optimized spend (million CU) | Change | Original mROAS | Optimized mROAS | Constraint |','|---|---:|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['channel']} | {r['original_spend']/1e6:.2f} | {r['optimized_spend']/1e6:.2f} | {r['change_pct']:+.1f}% | {r['original_marginal_roas']:.2f} | {r['optimized_marginal_roas']:.2f} | {r['bound']} |")
(OUT/'allocation.md').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
js=np.arange(5)
axes[0].bar(js-.18,spend/1e6,.36,label='Original',color='#94a3b8')
axes[0].bar(js+.18,spend*allocation/1e6,.36,label='Optimized',color='#0f766e')
axes[0].set(xticks=js,xticklabels=[f'Channel{i}' for i in js],ylabel='Spend (million currency units)',title='Same total budget; channel bounds ±20%');axes[0].legend(frameon=False)
axes[1].hist(gain.ravel()/1e6,bins=50,color='#0f766e',alpha=.8,density=True)
axes[1].axvline(0,color='#b91c1c',lw=1.5)
for q in np.quantile(gain,[.05,.95]):axes[1].axvline(q/1e6,color='#334155',ls='--')
axes[1].set(xlabel='Additional revenue vs original (million CU)',ylabel='Posterior density',title='One allocation, evaluated across 4,000 draws')
fig.tight_layout();fig.savefig(OUT/'budget_comparison.png',dpi=170);plt.close(fig)
print(json.dumps(meta,indent=2));print('\n'.join(lines))
