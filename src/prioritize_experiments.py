"""Decision-focused experiment screening under a finite posterior approximation.

No experimental observations are used. Quadrature averages over hypothetical
Normal summaries; likelihood reweighting updates the historical posterior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import prior_predictive as model

ROOT=Path(__file__).resolve().parents[1]


def solve(grid,curve,derivative,budget_weights,start=None):
    spline=CubicHermiteSpline(grid,curve,derivative,axis=0)
    j=np.arange(5)
    def utility(x):return float(spline(x)[j,j].sum())
    def grad(x):return spline(x,1)[j,j]
    r=minimize(lambda x:-utility(x),np.ones(5) if start is None else start,
               jac=lambda x:-grad(x),bounds=[(.8,1.2)]*5,
               constraints=[dict(type='eq',fun=lambda x:budget_weights@x-1,jac=lambda x:budget_weights)],
               method='SLSQP',options=dict(ftol=1e-11,maxiter=150))
    if not r.success:raise RuntimeError(r.message)
    assert abs(budget_weights@r.x-1)<1e-8
    return r.x,utility(r.x),spline


def build_cache(source,cache):
    model.ROOT=source
    p=model.prepare();n=p['cutoff'];lag=model.LAG
    raw=pd.read_csv(source/'data/geo_all_channels.csv').set_index(['geo','time']).reindex(pd.MultiIndex.from_product([p['geos'],p['dates']]))
    rpc=raw.revenue_per_conversion.to_numpy().reshape(p['y'].shape)
    spends=raw[[f'Channel{i}_spend' for i in range(5)]].to_numpy().reshape(*p['y'].shape,5)
    spend=spends[:,lag:n].sum(axis=(0,1))
    path=source/'reports/mcmc/posterior_samples.npz'
    with np.load(path) as f:
        decay=f['decay'].reshape(-1,5);k=f['k'].reshape(-1,5);beta=f['beta'].reshape(-1,len(p['geos']),5)
    grid=np.linspace(.8,1.2,41);count=len(decay)
    curves=np.empty((count,len(grid),5));derivatives=np.empty_like(curves)
    returns=np.empty((count,5))
    x=p['x'][:,:n];xp=x.copy();xp[:,lag:]=0
    mult=p['pop'][:,lag:n]*p['yscale']*rpc[:,lag:n]/1e6
    idx=np.arange(n-48,n-4);x0=p['x'][:,idx];pulse=np.zeros_like(x0);pulse[:,24:32]=.2*x0[:,24:32]
    emult=p['pop'][:,idx]*p['yscale']*rpc[:,idx]
    extra=.2*spends[:,idx][:,24:32].sum(axis=(0,1))
    for d in range(count):
        pre=model.adstock(xp,decay[d])[:,lag:]
        window=model.adstock(x,decay[d])[:,lag:]-pre
        v=window/(pre+k[d]);w=beta[d,:,None,:]*k[d]/(pre+k[d])*mult[:,:,None]
        for j,s in enumerate(grid):
            curves[d,j]=(w*v*s/(1+v*s)).sum(axis=(0,1))
            derivatives[d,j]=(w*v/(1+v*s)**2).sum(axis=(0,1))
        a=model.adstock(x0,decay[d]);inc=model.adstock(pulse,decay[d])
        effect=beta[d,:,None,:]*((a+inc)/(a+inc+k[d])-a/(a+k[d]))
        # Additivity permits computing five separate single-channel contrasts together.
        returns[d]=(effect[:,24:]*emult[:,24:,None]).sum(axis=(0,1))/extra
        if (d+1)%1000==0:print(f'Prepared {d+1}/{count} historical posterior draws',flush=True)
    # Reproduce the original decision and Channel2 intervention before using cache.
    allocation,_,_=solve(grid,curves.mean(0),derivatives.mean(0),spend/spend.sum())
    ref=pd.read_csv(source/'reports/budget_optimization/allocation.csv')
    np.testing.assert_allclose(allocation,ref.optimized_spend/ref.original_spend,atol=2e-5)
    refpath=source/'reports/experiment_results/uncalibrated_experiment_predictions.npz'
    if not refpath.exists():refpath=source/'reports/illustrative_calibration/uncalibrated_experiment_predictions.npz'
    if refpath.exists():
        with np.load(refpath) as f:np.testing.assert_allclose(returns[:,2],f['revenue_lift'].ravel()/extra[2],rtol=1e-9)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,grid=grid,curves=curves,derivatives=derivatives,returns=returns,spend=spend,extra=extra,
                        posterior_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        data_sha256=hashlib.sha256((source/'data/geo_all_channels.csv').read_bytes()).hexdigest())


def integrate(grid,curves,derivatives,ret,se,weights,nodes):
    current,_,_=solve(grid,curves.mean(0),derivatives.mean(0),weights)
    t,w=leggauss(nodes);lo=ret.min()-8*se;hi=ret.max()+8*se
    y=lo+(t+1)*(hi-lo)/2;qw=w*(hi-lo)/2
    loglike=norm.logpdf(y[:,None],ret[None,:],se)
    logz=logsumexp(loglike,axis=1)
    posterior=np.exp(loglike-logz[:,None]);mass=qw*np.exp(logz-np.log(len(ret)))
    assert abs(mass.sum()-1)<2e-5,(nodes,mass.sum())
    means=(posterior@curves.reshape(len(ret),-1)).reshape(nodes,len(grid),5)
    grads=(posterior@derivatives.reshape(len(ret),-1)).reshape(nodes,len(grid),5)
    ess=1/(posterior**2).sum(1);gains=[];moves=[];binding=[]
    for c,d in zip(means,grads):
        allocation,value,spline=solve(grid,c,d,weights,current)
        gain=value-spline(current)[np.arange(5),np.arange(5)].sum()
        assert gain>=-1e-7
        gains.append(max(0,gain))
        # Share of total budget moved (half the L1 allocation difference).
        moves.append(.5*np.sum(weights*np.abs(allocation-current)))
        binding.append(allocation[2]>1.19999)
    return dict(evsi_cu=float(mass@np.array(gains)*1e6),
                expected_budget_moved_pct=float(mass@np.array(moves)*100),
                probability_move_over_1pct=float(mass@(np.array(moves)>.01)),
                probability_channel2_at_upper_bound=float(mass@np.array(binding)),
                mean_weight_ess=float(mass@ess),predictive_mass_weight_ess_below_100=float(mass@(ess<100)),
                predictive_mass=float(mass.sum()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/experiment_prioritization')
    parser.add_argument('--cache',type=Path,default=ROOT/'runs/experiment_prioritization/cache.npz')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if not args.cache.exists():build_cache(args.model_root,args.cache)
    with np.load(args.cache) as f:a={k:f[k] for k in f.files}
    for relative,key in [('reports/mcmc/posterior_samples.npz','posterior_sha256'),('data/geo_all_channels.csv','data_sha256')]:
        actual=hashlib.sha256((args.model_root/relative).read_bytes()).hexdigest()
        if actual!=str(a[key]):raise ValueError('Cache inputs changed; rebuild the cache')
    grid=a['grid'];curves=a['curves'];derivatives=a['derivatives'];weights=a['spend']/a['spend'].sum()
    uninformative=integrate(grid,curves[:100],derivatives[:100],np.ones(100),.3,weights,128)
    assert uninformative['evsi_cu']<.01,uninformative
    allocation,value,_=solve(grid,curves.mean(0),derivatives.mean(0),weights)
    perfect=np.array([solve(grid,c,d,weights)[1] for c,d in zip(curves,derivatives)])
    evpi=float((perfect.mean()-value)*1e6)
    rows=[]
    for channel in range(5):
        for label,se in [('low',.60),('medium',.30),('high',.15)]:
            ret=a['returns'][:,channel]
            result=integrate(grid,curves,derivatives,ret,se,weights,256)
            refined=integrate(grid,curves,derivatives,ret,se,weights,512)
            halves=[]
            for indexes in [np.arange(0,len(ret)//2),np.arange(len(ret)//2,len(ret))]:
                halves.append(integrate(grid,curves[indexes],derivatives[indexes],ret[indexes],se,weights,256)['evsi_cu'])
            row=dict(channel=f'Channel{channel}',precision=label,se_return=se,
                     extra_experiment_spend_cu=float(a['extra'][channel]),
                     se_lift_cu=float(se*a['extra'][channel]),
                     current_return_mean=float(ret.mean()),current_return_sd=float(ret.std(ddof=1)),
                     **refined,quadrature_change_cu=abs(refined['evsi_cu']-result['evsi_cu']),
                     first_two_chains_evsi_cu=halves[0],last_two_chains_evsi_cu=halves[1])
            assert row['evsi_cu']<=evpi+1
            rows.append(row);print(f"{row['channel']} {label}: EVSI {row['evsi_cu']:,.0f} CU; quadrature change {row['quadrature_change_cu']:.2f}",flush=True)
    table=pd.DataFrame(rows)
    for _,group in table.groupby('channel'):
        assert np.all(np.diff(group.sort_values('se_return').evsi_cu)<=2),group
    table.to_csv(out/'candidate_results.csv',index=False)
    summary=dict(posterior='Historical-data posterior before hypothetical calibration',draws=len(curves),
                 budget_cu=float(a['spend'].sum()),decision_window_weeks=118,
                 decision_scenario='Replay the historical training allocation template; not a future forecast',
                 baseline_allocation_scale=allocation.tolist(),evpi_cu=evpi,
                 no_experiment_evsi_cu=0,precision_standard_errors_return=[.60,.30,.15],
                 candidate_intervention='One channel +20% for 8 weeks, 12 followup weeks, all 40 geos',
                 utility='Expected in-window revenue; excludes experiment costs, margins and delay costs',
                 method='512-node predictive quadrature and likelihood reweighting of 4000 MCMC draws; 256-node and two-chain-group sensitivity checks',
                 ess_note='Weight ESS measures concentration, not autocorrelation-adjusted MCMC ESS',
                 posterior_sha256=str(a['posterior_sha256']),data_sha256=str(a['data_sha256']),
                 max_quadrature_change_cu=float(table.quadrature_change_cu.max()),
                 max_predictive_mass_weight_ess_below_100=float(table.predictive_mass_weight_ess_below_100.max()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    colors=plt.cm.tab10(np.arange(5))
    for j,(channel,g) in enumerate(table.groupby('channel')):
        g=g.sort_values('se_return',ascending=False)
        axes[0].plot(g.se_return,g.evsi_cu/1e3,'o-',label=channel,color=colors[j])
    axes[0].invert_xaxis();axes[0].set(xticks=[.60,.30,.15],xlabel='Assumed return standard error (smaller = more precise)',ylabel='Expected information value (thousand CU)',title='Which experiment could improve allocation?')
    axes[0].legend(frameon=False,fontsize=9)
    g=table[table.precision=='medium'].sort_values('evsi_cu')
    axes[1].barh(g.channel,g.evsi_cu/1e3,color=[colors[int(c[-1])] for c in g.channel])
    for i,(_,r) in enumerate(g.iterrows()):axes[1].text(r.evsi_cu/1e3,i,f'  {r.evsi_cu/1e3:.1f}',va='center')
    axes[1].set(xlabel='Expected information value (thousand CU)',title='Comparison at return SE = 0.30')
    axes[1].set_xlim(0,g.evsi_cu.max()/1e3*1.25)
    fig.suptitle('Experiment prioritization under the historical MMM posterior',fontsize=13)
    fig.text(.5,.015,'Simulated business scenario • Gross revenue value over 118 weeks • Assumed precision, not achieved power',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(out/'experiment_value.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
