"""Frozen temporal conditional prediction, followed by first holdout scoring."""
import os,json,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prior_predictive import prepare,adstock,ROOT,LAG

OUT=ROOT/'reports/holdout'
SEED=20261002


def predict(p,samples):
    n=p['cutoff'];g,t=p['x'].shape[:2]
    chains,draws=samples['sigma'].shape
    rng=np.random.default_rng(SEED)
    means=np.empty((chains*draws,g,t-n));rep=np.empty_like(means)
    for c in range(chains):
        for d in range(draws):
            s={k:v[c,d] for k,v in samples.items()}
            a=adstock(p['x'],s['decay'])[:,n:]
            mu=s['alpha'][:,None]+p['basis'][n:]@s['delta']+p['z'][:,n:]@s['gamma']+(s['beta'][:,None,:]*a/(a+s['k'])).sum(axis=2)
            means[c*draws+d]=mu
            rep[c*draws+d]=mu+rng.normal(0,s['sigma'],mu.shape)
    return means,rep


def benchmarks(p):
    n=p['cutoff'];t=len(p['dates']);h=t-n
    assert t-1-52<n
    return {'Regional mean':np.repeat(p['y_scaled'][:,LAG:n].mean(axis=1)[:,None],h,axis=1),
            'Seasonal naive':p['y_scaled'][:,np.arange(n,t)-52].copy()}


def scores(obs,pred,mult):
    err=pred-obs;actual=obs*mult;original=pred*mult
    aggerr=original.sum(axis=0)-actual.sum(axis=0)
    return {'rmse_scaled':float(np.sqrt(np.mean(err**2))),'mae_scaled':float(np.mean(abs(err))),
            'rmse_original':float(np.sqrt(np.mean((original-actual)**2))),
            'mae_original':float(np.mean(abs(original-actual))),
            'wape':float(abs(original-actual).sum()/actual.sum()),'bias_scaled':float(err.mean()),
            'aggregate_weekly_rmse':float(np.sqrt(np.mean(aggerr**2))),
            'aggregate_weekly_wape':float(abs(aggerr).sum()/actual.sum())}


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'metrics.json').exists():
        raise RuntimeError('First-evaluation outputs already exist. Preserve them; do not overwrite silently.')
    paths=['HOLDOUT_PROTOCOL.md','src/fit_mcmc.py','src/prior_predictive.py','src/holdout_prediction.py','data/geo_all_channels.csv','reports/mcmc/posterior_samples.npz']
    manifest={'seed':SEED,'evaluation':'first temporal holdout, frozen model','files_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in paths}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    p=prepare();s=dict(np.load(ROOT/'reports/mcmc/posterior_samples.npz'))
    mu,rep=predict(p,s);base=benchmarks(p)
    # Perturb unseen Y: neither predicted outcomes nor baselines may change.
    altered={**p,'y_scaled':p['y_scaled'].copy(),'y':p['y'].copy()}
    altered['y_scaled'][:,p['cutoff']:]*=1000;altered['y'][:,p['cutoff']:]*=1000
    tiny={k:v[:1,:1] for k,v in s.items()}
    check,_=predict(altered,tiny)
    assert np.allclose(check[0],mu[0])
    for k,v in benchmarks(altered).items():assert np.array_equal(v,base[k])
    # In particular, altering a future exposure cannot change earlier adstock.
    altered_x=p['x'].copy();altered_x[:,-1]*=100
    assert np.array_equal(adstock(altered_x,s['decay'][0,0])[:,:-1],adstock(p['x'],s['decay'][0,0])[:,:-1])
    n=p['cutoff'];mult=p['pop'][:,n:]*p['yscale'];point=mu.mean(axis=0)
    levels=[.5,.8,.9,.95]
    qs={level:np.quantile(rep,[(1-level)/2,1-(1-level)/2],axis=0) for level in levels}
    totals=(rep*mult).sum(axis=1);tq=np.quantile(totals,[.05,.95],axis=0)
    np.savez_compressed(OUT/'predictions_before_scoring.npz',mean_scaled=point,interval90_scaled=qs[.9],total_interval90=tq,
                        regional_mean_scaled=base['Regional mean'],seasonal_naive_scaled=base['Seasonal naive'])
    # First access to held-out outcomes for evaluation is below.
    obs=p['y_scaled'][:,n:];obs_original=p['y'][:,n:]
    models={'MMM':point,**base}
    metrics={name:scores(obs,pred,mult) for name,pred in models.items()}
    coverage=[]
    for level in levels:
        lo,hi=qs[level];coverage.append({'nominal':level,'coverage':float(((obs>=lo)&(obs<=hi)).mean()),'mean_width_scaled':float((hi-lo).mean())})
    lo,hi=qs[.9];alpha=.1
    interval_score=hi-lo+2/alpha*(lo-obs)*(obs<lo)+2/alpha*(obs-hi)*(obs>hi)
    obs_total=obs_original.sum(axis=0)
    result={'metrics':metrics,'coverage':coverage,'cells':int(obs.size),'regions':len(p['geos']),'weeks':len(p['dates'])-n,
            'start':p['dates'][n],'end':p['dates'][-1],'posterior_draws':len(rep),
            'mean_90pct_interval_score_scaled':float(interval_score.mean()),
            'weekly_aggregate_90pct_coverage':float(((obs_total>=tq[0])&(obs_total<=tq[1])).mean()),
            'weekly_aggregate_90pct_mean_width':float((tq[1]-tq[0]).mean()),
            'negative_expected_fraction':float((mu<0).mean()),'negative_predictive_fraction':float((rep<0).mean()),
            'checks':{'holdout_outcomes_do_not_affect_predictions':True,'no_future_media_leakage':True,'all_seasonal_reference_outcomes_in_training':True},
            'note':'First frozen-model conditional holdout prediction using actual contemporaneous inputs. No refitting. Not causal validation or an advance forecast.'}
    regions=[]
    for i,geo in enumerate(p['geos']):
        row={'geo':geo,'coverage90':float(((obs[i]>=lo[i])&(obs[i]<=hi[i])).mean()),'interval_width90_scaled':float((hi[i]-lo[i]).mean())}
        for name,pred in models.items():row[name+'_rmse_scaled']=float(np.sqrt(np.mean((pred[i]-obs[i])**2)))
        regions.append(row)
    regional=pd.DataFrame(regions);regional.to_csv(OUT/'region_metrics.csv',index=False)
    result['regions_mmm_beats_regional_mean']=int((regional.MMM_rmse_scaled<regional['Regional mean_rmse_scaled']).sum())
    result['regions_mmm_beats_seasonal_naive']=int((regional.MMM_rmse_scaled<regional['Seasonal naive_rmse_scaled']).sum())
    support=[]
    for kind,arr,names in [('media',p['x'],[f'Channel{i}' for i in range(5)]),('control',p['z'],['competitor','sentiment','Promo','organic'])]:
        lower=arr[:,LAG:n].min(axis=1);upper=arr[:,LAG:n].max(axis=1)
        outside=(arr[:,n:]<lower[:,None,:])|(arr[:,n:]>upper[:,None,:])
        for j,name in enumerate(names):support.append({'kind':kind,'variable':name,'fraction_outside_own_region_training_range':float(outside[:,:,j].mean())})
    pd.DataFrame(support).to_csv(OUT/'input_support.csv',index=False)
    result['input_support']=support
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(metrics).T.to_csv(OUT/'model_comparison.csv')
    rows=[]
    for g,geo in enumerate(p['geos']):
        for j,date in enumerate(p['dates'][n:]):
            rows.append({'geo':geo,'time':date,'observed':obs_original[g,j],**{name:pred[g,j]*mult[g,j] for name,pred in models.items()},'lo90':lo[g,j]*mult[g,j],'hi90':hi[g,j]*mult[g,j]})
    pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
    plot(p,result,models,obs,mult,tq,regional)
    print(json.dumps(result,indent=2))


def plot(p,result,models,obs,mult,tq,regional):
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    colors={'MMM':'#4263b0','Regional mean':'#b35431','Seasonal naive':'#42896c'}
    dates=pd.to_datetime(p['dates'][p['cutoff']:])
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    ax=axes[0];ax.fill_between(dates,tq[0]/1e6,tq[1]/1e6,color=colors['MMM'],alpha=.2,label='MMM 90% predictive interval')
    ax.plot(dates,(obs*mult).sum(axis=0)/1e6,color='#20252b',label='Observed',lw=2)
    for name,pred in models.items():ax.plot(dates,(pred*mult).sum(axis=0)/1e6,color=colors[name],label=name,ls='-' if name=='MMM' else '--')
    ax.set(title='26-week holdout: conditional predictions with actual inputs',ylabel='Total conversions (millions)');ax.legend(fontsize=8,ncol=2)
    for name,pred in models.items():axes[1].plot(dates,np.sqrt(((pred-obs)**2).mean(axis=0)),color=colors[name],label=name)
    axes[1].set(title='Weekly region-level error (equal region weights)',ylabel='RMSE on scaled per-capita KPI',xlabel='Holdout week');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'holdout_predictions.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    c=result['coverage'];axes[0].plot([0,1],[0,1],color='grey',ls='--',label='Nominal = observed')
    axes[0].plot([v['nominal'] for v in c],[v['coverage'] for v in c],'o-',color=colors['MMM'])
    axes[0].set(xlim=(.45,1),ylim=(.45,1),xlabel='Nominal coverage',ylabel='Observed holdout coverage',title='Predictive interval coverage')
    from matplotlib.ticker import PercentFormatter
    axes[0].xaxis.set_major_formatter(PercentFormatter(1));axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for name in ['Regional mean','Seasonal naive']:
        vals=regional[name+'_rmse_scaled'];axes[1].scatter(vals,regional.MMM_rmse_scaled,label=name,color=colors[name],s=24,alpha=.8)
    bound=max(regional.filter(like='rmse_scaled').max())*1.05
    axes[1].plot([0,bound],[0,bound],color='grey',ls='--');axes[1].set(xlim=(0,bound),ylim=(0,bound),xlabel='Benchmark regional RMSE',ylabel='MMM regional RMSE',title='All 40 regions; below diagonal favors MMM');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'holdout_calibration_regions.png');plt.close(fig)

if __name__=='__main__':main()
