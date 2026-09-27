"""Readouts for the completed hypothetical-evidence calibration workflow."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prior_predictive import ROOT,LAG,prepare,adstock

OUT=ROOT/'reports/illustrative_calibration'
old_dir=ROOT/'reports/budget_optimization'
new_dir=ROOT/'reports/calibrated_budget'
e=json.loads((ROOT/'data/illustrative_experiment/evidence.json').read_text())
diag=json.loads((OUT/'diagnostics.json').read_text())
assert diag['divergences']==0 and diag['max_rhat']<1.01 and diag['min_bulk_ess']>400,diag
with np.load(ROOT/'reports/illustrative_calibration/uncalibrated_experiment_predictions.npz') as f:old_return=f['revenue_lift']/e['full_population_incremental_spend']
with np.load(OUT/'posterior_samples.npz') as f:
    samples={k:f[k] for k in ['experiment_return','experiment_lift','beta','decay','k']}
new_return=samples['experiment_return']
def stats(v):
    return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),lo90=float(np.quantile(v,.05)),hi90=float(np.quantile(v,.95)))
old_channels=pd.read_csv(ROOT/'reports/channel_results/channel_results.csv')
new_channels=pd.read_csv(ROOT/'reports/calibrated_channels/channel_results.csv')
channels=old_channels.merge(new_channels,on='channel',suffixes=('_before','_after'))
channels.to_csv(OUT/'channel_comparison.csv',index=False)
old_alloc=pd.read_csv(old_dir/'allocation.csv');new_alloc=pd.read_csv(new_dir/'allocation.csv')
allocation=old_alloc[['channel','original_spend','optimized_spend','change_pct']].merge(new_alloc[['channel','optimized_spend','change_pct']],on='channel',suffixes=('_before','_after'))
allocation.to_csv(OUT/'allocation_comparison.csv',index=False)
# Compare the two policies under the SAME calibrated posterior.
p=prepare();n=p['cutoff'];sl=slice(LAG,n)
raw=pd.read_csv(ROOT/'data/geo_all_channels.csv').set_index(['geo','time']).reindex(pd.MultiIndex.from_product([p['geos'],p['dates']]))
rpc=raw.revenue_per_conversion.to_numpy().reshape(p['y'].shape)[:,sl]
mult=p['pop'][:,sl]*p['yscale']*rpc
x=p['x'][:,:n].copy();xp=x.copy();xp[:,LAG:]=0
old_scale=old_alloc.optimized_spend.to_numpy()/old_alloc.original_spend.to_numpy()
old_policy=np.empty(new_return.shape)
for c in range(new_return.shape[0]):
    for d in range(new_return.shape[1]):
        decay=samples['decay'][c,d];k=samples['k'][c,d];beta=samples['beta'][c,d]
        a=adstock(x,decay)[:,sl];pre=adstock(xp,decay)[:,sl]
        new_a=pre+(a-pre)*old_scale
        old_policy[c,d]=((beta[:,None,:]*(new_a/(new_a+k)-pre/(pre+k))).sum(axis=2)*mult).sum()
with np.load(new_dir/'posterior_comparison.npz') as f:
    optimized=f['optimized_revenue'].sum(axis=2);baseline=f['baseline_revenue'].sum(axis=2)
    new_gain=f['revenue_gain']
policy_gain=optimized-old_policy
np.savez_compressed(OUT/'policy_comparison_same_posterior.npz',old_policy_revenue=old_policy,new_policy_revenue=optimized,new_minus_old_policy=policy_gain)
assert policy_gain.mean()>-1e-5
old_budget=json.loads((old_dir/'summary.json').read_text());new_budget=json.loads((new_dir/'summary.json').read_text())
summary=dict(label='Hypothetical independent experimental evidence; not measured or derived from v1',
    experiment=e,experiment_return_before=stats(old_return),experiment_return_after=stats(new_return),
    experiment_return_sd_reduction_pct=100*(1-new_return.std()/old_return.std()),
    old_optimization_under_old_posterior=old_budget['revenue_gain'],
    new_optimization_under_calibrated_posterior=new_budget['revenue_gain'],
    new_vs_old_policy_under_same_calibrated_posterior=stats(policy_gain),
    probability_new_policy_better_under_calibrated_posterior=float((policy_gain>0).mean()),
    diagnostics=diag,causal_validation_claimed=False,
    interpretation='Posterior change is conditional on the hypothetical evidence and its assumed precision and independence. Experiment return differs from historical channel ROAS and marginal ROAS.')
(OUT/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
# Density curves are fitted KDEs of posterior samples; experimental curve is likelihood.
from scipy.stats import gaussian_kde
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
grid=np.linspace(0,max(np.quantile(old_return,.995),2.5),500)
axes[0].plot(grid,gaussian_kde(old_return.ravel())(grid),label='Before calibration',color='#64748b',lw=2)
axes[0].plot(grid,norm.pdf(grid,1.2,.15),label='Hypothetical experiment likelihood',color='#d97706',ls='--',lw=2)
axes[0].plot(grid,gaussian_kde(new_return.ravel())(grid),label='After calibration',color='#0f766e',lw=2)
axes[0].set(xlabel='Return for the exact experimental intervention',ylabel='Density (likelihood scaled for display)',title='New evidence updates the posterior');axes[0].legend(frameon=False,fontsize=9)
js=np.arange(5)
axes[1].bar(js-.18,allocation.optimized_spend_before/1e6,.36,label='Before calibration',color='#94a3b8')
axes[1].bar(js+.18,allocation.optimized_spend_after/1e6,.36,label='After calibration',color='#0f766e')
axes[1].set(xticks=js,xticklabels=allocation.channel,xlabel='Same fixed total budget and ±20% bounds',ylabel='Optimized spend (million CU)',title='Optimal allocation barely changes');axes[1].legend(frameon=False)
channel2_index=allocation.channel.tolist().index('Channel2')
channel2_spend=allocation.iloc[channel2_index].optimized_spend_after/1e6
axes[1].annotate('Channel2 stays at the +20% cap\nLower estimated return, same budget limit',
                 xy=(channel2_index,channel2_spend),xytext=(.04,.57),textcoords='axes fraction',
                 fontsize=9,color='#334155',ha='left',va='center',
                 bbox=dict(boxstyle='round,pad=0.45',fc='#f1f5f9',ec='none'),
                 arrowprops=dict(arrowstyle='->',color='#64748b',connectionstyle='angle3,angleA=0,angleB=90'))
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('MMM calibration with hypothetical experimental evidence',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'calibration_workflow.png',dpi=180);plt.close(fig)

def fmt(row,metric,suffix):return f"{row[f'{metric}_mean_{suffix}']:.2f} [{row[f'{metric}_lo90_{suffix}']:.2f}, {row[f'{metric}_hi90_{suffix}']:.2f}]"
lines=['| Channel | ROAS before | ROAS after | Marginal ROAS before | Marginal ROAS after |','|---|---:|---:|---:|---:|']
for _,r in channels.iterrows():lines.append('| '+' | '.join([r.channel,fmt(r,'roas','before'),fmt(r,'roas','after'),fmt(r,'marginal_roas','before'),fmt(r,'marginal_roas','after')])+' |')
(OUT/'channel_comparison.md').write_text('\n'.join(lines)+'\n')
lines=['| Channel | Original spend (million CU) | Optimized before calibration | Optimized after calibration | Change from original after calibration |','|---|---:|---:|---:|---:|']
for _,r in allocation.iterrows():lines.append(f"| {r.channel} | {r.original_spend/1e6:.2f} | {r.optimized_spend_before/1e6:.2f} | {r.optimized_spend_after/1e6:.2f} | {r.change_pct_after:+.1f}% |")
(OUT/'allocation_comparison.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2));print('\n'.join(lines))
