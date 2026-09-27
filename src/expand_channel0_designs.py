"""Compare intervention intensity at equal assumed revenue-summary noise."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import prior_predictive as model
from prioritize_experiments import ROOT,build_cache,integrate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-root',type=Path,default=ROOT)
    parser.add_argument('--cache',type=Path,default=ROOT/'runs/experiment_prioritization/cache.npz')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/channel0_designs')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if not args.cache.exists():build_cache(args.model_root,args.cache)
    with np.load(args.cache) as f:a={k:f[k] for k in f.files}
    for relative,key in [('reports/mcmc/posterior_samples.npz','posterior_sha256'),('data/geo_all_channels.csv','data_sha256')]:
        assert hashlib.sha256((args.model_root/relative).read_bytes()).hexdigest()==str(a[key]),'Stale cache'
    model.ROOT=args.model_root;p=model.prepare();n=p['cutoff'];idx=np.arange(n-48,n-4)
    raw=pd.read_csv(args.model_root/'data/geo_all_channels.csv').set_index(['geo','time']).reindex(pd.MultiIndex.from_product([p['geos'],p['dates']]))
    rpc=raw.revenue_per_conversion.to_numpy().reshape(p['y'].shape)[:,idx]
    multiplier=p['pop'][:,idx]*p['yscale']*rpc
    x0=p['x'][:,idx].copy();pulse=np.zeros_like(x0);pulse[:,24:32,0]=x0[:,24:32,0]
    with np.load(args.model_root/'reports/mcmc/posterior_samples.npz') as f:
        s={k:f[k].reshape((-1,)+f[k].shape[2:]) for k in ['beta','decay','k']}
    intensities=[-.2,.1,.2];lifts=np.empty((len(a['curves']),3))
    for d in range(len(lifts)):
        base=model.adstock(x0,s['decay'][d]);change=model.adstock(pulse,s['decay'][d])
        for j,intensity in enumerate(intensities):
            treated=base+intensity*change
            effect=s['beta'][d,:,None,:]*(treated/(treated+s['k'][d])-base/(base+s['k'][d]))
            lifts[d,j]=(effect[:,24:,0]*multiplier[:,24:]).sum()
    reference_spend=float(a['extra'][0])
    np.testing.assert_allclose(lifts[:,2]/reference_spend,a['returns'][:,0],rtol=1e-9)
    assert (lifts[:,0]<0).all() and (lifts[:,1:]>0).all()
    # Concavity implies larger losses for -20% than gains for +20%, draw by draw.
    assert (-lifts[:,0]>=lifts[:,2]-1e-6).all()
    assert (lifts[:,1]>=.5*lifts[:,2]-1e-6).all()
    weights=a['spend']/a['spend'].sum();rows=[]
    for j,intensity in enumerate(intensities):
        spend=reference_spend*intensity/.2
        # All designs share a positive unit conversion; negative outcomes stay negative.
        signal=lifts[:,j]/reference_spend
        for label,se in [('low',.6),('medium',.3),('high',.15)]:
            coarse=integrate(a['grid'],a['curves'],a['derivatives'],signal,se,weights,256)
            fine=integrate(a['grid'],a['curves'],a['derivatives'],signal,se,weights,512)
            halves=[]
            for ix in [np.arange(2000),np.arange(2000,4000)]:
                halves.append(integrate(a['grid'],a['curves'][ix],a['derivatives'][ix],signal[ix],se,weights,256)['evsi_cu'])
            row=dict(intensity_pct=intensity*100,precision=label,spend_change_cu=spend,
                     revenue_change_mean_cu=float(lifts[:,j].mean()),revenue_change_lo90_cu=float(np.quantile(lifts[:,j],.05)),
                     revenue_change_hi90_cu=float(np.quantile(lifts[:,j],.95)),se_revenue_cu=se*reference_spend,
                     se_per_absolute_spend=se*reference_spend/abs(spend),
                     **fine,quadrature_change_cu=abs(fine['evsi_cu']-coarse['evsi_cu']),
                     first_two_chains_evsi_cu=halves[0],last_two_chains_evsi_cu=halves[1])
            rows.append(row);print(f"{intensity:+.0%} {label}: {fine['evsi_cu']:,.0f} CU",flush=True)
    t=pd.DataFrame(rows)
    old=pd.read_csv(ROOT/'reports/experiment_prioritization/candidate_results.csv')
    for label in ['low','medium','high']:
        original=float(old[(old.channel=='Channel0')&(old.precision==label)].evsi_cu.iloc[0])
        actual=float(t[(t.intensity_pct==20)&(t.precision==label)].evsi_cu.iloc[0])
        np.testing.assert_allclose(actual,original,atol=1)
    for _,g in t.groupby('intensity_pct'):
        assert np.all(np.diff(g.sort_values('se_revenue_cu').evsi_cu)<=2)
    t.to_csv(out/'design_comparison.csv',index=False)
    summary=dict(channel='Channel0',intensities_pct=[-20,10,20],posterior_draws=len(lifts),
                 experiment_weeks=8,followup_weeks=12,decision_window_weeks=118,
                 reference_plus20_spend_cu=reference_spend,
                 common_revenue_standard_errors_cu=[reference_spend*x for x in [.6,.3,.15]],
                 likelihood='Normal(signed incremental revenue, common assumed revenue SE)',
                 negative_intervention='Revenue change and spend change are both signed treatment-minus-control differences; negative revenue is a loss, negative spend a saving',
                 utility='Gross expected downstream revenue decision value; excludes experiment revenue/costs, margins and delay',
                 caveat='Equal revenue noise is a comparison assumption, not demonstrated equal feasibility or experiment cost',
                 max_quadrature_change_cu=float(t.quadrature_change_cu.max()),
                 max_low_weight_ess_probability=float(t.predictive_mass_weight_ess_below_100.max()),
                 posterior_sha256=str(a['posterior_sha256']),data_sha256=str(a['data_sha256']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for intensity,g in t.groupby('intensity_pct'):
        g=g.sort_values('se_revenue_cu',ascending=False)
        axes[0].plot(g.se_revenue_cu/1e3,g.evsi_cu/1e6,'o-',label=f'{intensity:+.0f}% spend')
    axes[0].invert_xaxis();axes[0].legend(frameon=False)
    axes[0].set(xlabel='Common revenue SE (thousand CU; smaller = more precise)',ylabel='Expected information value (million CU)',title='Equal revenue-measurement uncertainty')
    mid=t[t.precision=='medium'].sort_values('intensity_pct')
    axes[1].bar([f'{x:+.0f}%' for x in mid.intensity_pct],mid.evsi_cu/1e6,color=['#4263b0','#94a3b8','#0f766e'])
    for j,v in enumerate(mid.evsi_cu/1e6):axes[1].text(j,v+.025,f'{v:.3f}',ha='center')
    axes[1].set(xlabel='Channel0 intervention',ylabel='Expected information value (million CU)',title=f'Medium precision: revenue SE = {reference_spend*.3/1000:.1f}k CU',ylim=(0,mid.evsi_cu.max()/1e6*1.2))
    fig.suptitle('Channel0: compare a spending cut with two increases',fontsize=13)
    fig.text(.5,.015,'Simulated scenario • 8-week intervention + 12-week followup • Downstream value over 118 weeks, before test costs',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(out/'intensity_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
