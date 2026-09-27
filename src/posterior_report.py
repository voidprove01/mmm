"""Diagnostics and training posterior predictive checks; never open holdout outcomes."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.matplotlib'))
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import arviz as az
from prior_predictive import ROOT,prepare,adstock,LAG


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--name',default='mcmc');opts=cli.parse_args()
    out=ROOT/'reports'/opts.name
    p=prepare();n=p['cutoff'];samples=dict(np.load(out/'posterior_samples.npz'))
    chains,draws=samples['sigma'].shape
    rng=np.random.default_rng(20261001)
    chosen=set((c,d) for c in range(chains) for d in np.linspace(0,draws-1,min(250,draws),dtype=int))
    pp=[];mu_sum=np.zeros_like(p['y_scaled'][:,LAG:n]);negmu=0
    paid_mean=np.empty((chains,draws,5))
    for c in range(chains):
        for d in range(draws):
            s={k:v[c,d] for k,v in samples.items()}
            a=adstock(p['x'][:,:n],s['decay'])[:,LAG:]
            paid=s['beta'][:,None,:]*a/(a+s['k'])
            mu=s['alpha'][:,None]+p['basis'][LAG:n]@s['delta']+p['z'][:,LAG:n]@s['gamma']+paid.sum(axis=2)
            mu_sum+=mu;negmu+=(mu<0).sum();paid_mean[c,d]=paid.mean(axis=(0,1))
            if (c,d) in chosen:pp.append(mu+rng.normal(0,s['sigma'],mu.shape))
    pp=np.array(pp);mu_bar=mu_sum/(chains*draws)
    observed=p['y_scaled'][:,LAG:n];q=np.quantile(pp,[.05,.5,.95],axis=0)
    resid=observed-mu_bar
    acf=[np.corrcoef(row[:-1],row[1:])[0,1] for row in resid]
    derived=az.from_dict(posterior={'mean_paid_contribution':paid_mean})
    ds=az.summary(derived,round_to=6);ds.to_csv(out/'derived_diagnostics.csv')
    totals=(pp*p['pop'][None,:,LAG:n]*p['yscale']).sum(axis=1)
    tq=np.quantile(totals,[.05,.5,.95],axis=0)/1e6
    dates=pd.to_datetime(p['dates'][LAG:n])
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    axes[0].fill_between(dates,tq[0],tq[2],color='#4263b0',alpha=.25,label='90% posterior predictive interval')
    axes[0].plot(dates,tq[1],color='#4263b0',label='Predictive median')
    axes[0].plot(dates,p['y'][:,LAG:n].sum(axis=0)/1e6,color='#20252b',lw=1,label='Observed training total')
    axes[0].set(title='Training posterior predictive check (not out-of-sample accuracy)',ylabel='Total conversions (millions)');axes[0].legend(fontsize=8)
    axes[1].plot(dates,resid.mean(axis=0),color='#4263b0');axes[1].axhline(0,color='grey',lw=.7)
    axes[1].set(title='Mean residual across regions',xlabel='Week',ylabel='Scaled outcome residual')
    fig.tight_layout();fig.savefig(out/'posterior_predictive.png');plt.close(fig)
    selected=[('alpha0',None),('tau_alpha',None),('sigma',None),('decay',0),('k',0),('tau',0)]
    fig,axes=plt.subplots(3,2,figsize=(11,7))
    for ax,(name,idx) in zip(axes.ravel(),selected):
        v=samples[name] if idx is None else samples[name][:,:,idx]
        for c in range(chains):ax.plot(v[c],lw=.6,alpha=.7,label=f'Chain {c+1}')
        ax.set(title=name if idx is None else f'{name}[Channel0]',xlabel='Retained draw',ylabel='Parameter value')
    axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle('Selected trace plots — full diagnostics cover all parameters');fig.tight_layout();fig.savefig(out/'trace_plots.png');plt.close(fig)
    priors={'decay':rng.beta(2,2,100000),'k':rng.lognormal(0,.6,100000),'tau':abs(rng.normal(0,.3,100000))}
    fig,axes=plt.subplots(1,3,figsize=(11,4.6))
    for ax,key in zip(axes,priors):
        pr=np.quantile(priors[key],[.05,.5,.95]);vals=samples[key].reshape(-1,5);pq=np.quantile(vals,[.05,.5,.95],axis=0)
        ax.axvspan(pr[0],pr[2],color='grey',alpha=.17,label='Prior 5–95%')
        ax.axvline(pr[1],color='grey',ls='--',label='Prior median')
        for m in range(5):ax.plot([pq[0,m],pq[2,m]],[m,m],color='#4263b0',lw=3);ax.scatter(pq[1,m],m,color='#4263b0',s=20)
        ax.set_yticks(range(5),[f'Channel{i}' for i in range(5)]);ax.set(title=key,xlabel='Parameter value');ax.invert_yaxis()
    axes[0].legend(fontsize=7);fig.suptitle('Prior versus posterior: medians and 90% equal-tail intervals');fig.tight_layout();fig.savefig(out/'prior_vs_posterior.png');plt.close(fig)
    metrics={'predictive_draws':len(pp),'posterior_negative_expected_cell_fraction':float(negmu/(mu_bar.size*chains*draws)),'posterior_negative_predictive_cell_fraction':float((pp<0).mean()),'training_cell_90pct_predictive_coverage':float(((observed>=q[0])&(observed<=q[2])).mean()),'training_rmse_scaled':float(np.sqrt(np.mean(resid**2))),'median_region_residual_lag1_correlation':float(np.median(acf)),'derived_max_rhat':float(ds.r_hat.max()),'derived_min_bulk_ess':float(ds.ess_bulk.min()),'derived_min_tail_ess':float(ds.ess_tail.min()),'note':'Training checks only. Posterior predictive uses up to 250 evenly spaced draws per chain. No holdout performance or causal validity claimed.'}
    (out/'predictive_checks.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
    np.savez_compressed(out/'predictive_summary.npz',mu_mean=mu_bar,predictive_quantiles=q,total_quantiles=tq,mean_paid_contribution=paid_mean)

if __name__=='__main__':main()
