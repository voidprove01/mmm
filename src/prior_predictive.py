"""Hierarchical MMM prior predictive simulation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260925
DRAWS = 1000
LAG = 12
HOLDOUT = 26


def prepare(path=None):
    df = pd.read_csv(path or ROOT / 'data/geo_all_channels.csv')
    dates = np.sort(df.time.unique())
    geos = sorted(df.geo.unique())
    df = df.set_index(['geo', 'time']).reindex(pd.MultiIndex.from_product([geos, dates], names=['geo','time']))
    g,t = len(geos),len(dates)
    def cube(cols):
        return df[cols].to_numpy().copy().reshape(g,t,len(cols))
    pop = cube(['population'])[:,:,0]
    if not np.all(pop>0): raise ValueError('Population must be positive')
    y = cube(['conversions'])[:,:,0]
    cutoff = t-HOLDOUT
    yscale = (y[:,:cutoff]/pop[:,:cutoff]).mean()
    media = cube([f'Channel{i}_impression' for i in range(5)])/pop[:,:,None]
    scales = np.array([np.median(media[:,:cutoff,m][media[:,:cutoff,m]>0]) for m in range(5)])
    x = media/scales
    controls = cube(['competitor_sales_control','sentiment_score_control','Promo','Organic_channel0_impression'])
    controls[:,:,3] /= pop
    center = controls[:,:cutoff].mean(axis=(0,1))
    sd = controls[:,:cutoff].std(axis=(0,1))
    if np.any(sd==0): raise ValueError('Constant control')
    z = (controls-center)/sd
    weeks = np.arange(t)
    basis = np.column_stack([(weeks-(cutoff-1)/2)/cutoff,
                             np.sin(2*np.pi*weeks/52), np.cos(2*np.pi*weeks/52)])
    return dict(y=y,pop=pop,y_scaled=y/pop/yscale,x=x,z=z,basis=basis,
                dates=dates,geos=geos,cutoff=cutoff,yscale=yscale,media_scales=scales,
                control_center=center,control_sd=sd)


def adstock(x, decay, lag=LAG):
    # Full, constant denominator. Unobserved prehistory treated as zero;
    # first lag weeks are excluded from all fitting/check summaries.
    out=np.zeros_like(x)
    weights=decay[None,:]**np.arange(lag+1)[:,None]
    weights/=weights.sum(axis=0)
    for j,w in enumerate(weights):
        out[:,j:,:]+=x[:,:x.shape[1]-j,:]*w
    return out


def simulate(p, draws=DRAWS, seed=SEED):
    rng=np.random.default_rng(seed)
    g,t,m=p['x'].shape
    n=p['cutoff']
    expected=np.empty((draws,n-LAG)); predicted=np.empty_like(expected)
    geo_pred=np.empty((draws,g)); neg=0;total=0;above=0
    medshare=[]; lambdas=[]; ks=[]; betas=[]
    components=np.empty((draws,g,n-LAG,5))
    mu_cells=np.empty((draws,g,n-LAG))
    y_cells=np.empty_like(mu_cells)
    # Training window only: holdout outcomes do not enter these checks.
    x=p['x'][:,:n]; z=p['z'][:,:n]; basis=p['basis'][:n]
    for d in range(draws):
        alpha0=rng.normal(.6,.2)
        tau_alpha=abs(rng.normal(0,.2))
        alpha=alpha0+tau_alpha*rng.normal(size=g)
        b=rng.normal(np.log(.15),.5,size=m)
        tau=abs(rng.normal(0,.3,size=m))
        beta=np.exp(b+tau*rng.normal(size=(g,m)))
        decay=rng.beta(2,2,size=m)
        k=rng.lognormal(0,.6,size=m)
        gamma=rng.normal(0,.1,size=4)
        delta=rng.normal(0,[.2,.1,.1])
        sigma=abs(rng.normal(0,.15))
        a=adstock(x,decay)
        paid=(beta[:,None,:]*a/(a+k)).sum(axis=2)
        mu=alpha[:,None]+basis@delta+z@gamma+paid
        yp=mu+rng.normal(0,sigma,size=mu.shape)
        sl=slice(LAG,n)
        components[d,:,:,0]=alpha[:,None]
        components[d,:,:,1]=paid[:,sl]
        components[d,:,:,2]=(z@gamma)[:,sl]
        components[d,:,:,3]=(basis@delta)[None,sl]
        components[d,:,:,4]=(yp-mu)[:,sl]
        mu_cells[d]=mu[:,sl]
        y_cells[d]=yp[:,sl]
        # Convert scaled per-capita predictions back to the original KPI.
        multiplier=p['pop'][:,sl]*p['yscale']
        expected[d]=(mu[:,sl]*multiplier).sum(axis=0)
        predicted[d]=(yp[:,sl]*multiplier).sum(axis=0)
        geo_pred[d]=yp[:,sl].mean(axis=1)
        neg+=(yp[:,sl]<0).sum();total+=yp[:,sl].size
        above+=(yp[:,sl]>3).sum()
        medshare.append(paid[:,sl].mean());lambdas.append(decay);ks.append(k);betas.append(beta)
    return dict(components=components,mu_cells=mu_cells,y_cells=y_cells,
                expected=expected,predicted=predicted,geo_pred=geo_pred,
                negative_fraction=float(neg/total),above_three_fraction=float(above/total),
                paid_component=np.array(medshare),decay=np.array(lambdas),k=np.array(ks),beta=np.array(betas))


def make_report(p,r):
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    n=p['cutoff']; dates=pd.to_datetime(p['dates'][LAG:n])
    fig,ax=plt.subplots(figsize=(10,4.5))
    q=np.quantile(r['predicted'],[.05,.5,.95],axis=0)
    ax.fill_between(dates,q[0]/1e6,q[2]/1e6,alpha=.22,color='#4263b0',label='90% prior predictive interval')
    ax.plot(dates,q[1]/1e6,color='#4263b0',label='Prior predictive median')
    ax.plot(dates,p['y'][:,LAG:n].sum(axis=0)/1e6,color='#20252b',label='Observed training KPI')
    ax.axhline(0,color='grey',lw=.6)
    ax.set(title='Before fitting: can the prior generate plausible total conversions?',ylabel='Total conversions (millions)',xlabel='Week (training only; first 12 weeks excluded)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'prior_predictive_total.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    obs=p['y_scaled'][:,LAG:n].mean(axis=1);order=np.argsort(obs)
    q=np.quantile(r['geo_pred'],[.05,.5,.95],axis=0)[:,order]
    xx=np.arange(len(order));ax.vlines(xx,q[0],q[2],color='#4263b0',alpha=.55,label='90% prior interval: regional mean')
    ax.scatter(xx,q[1],s=12,color='#4263b0');ax.scatter(xx,obs[order],s=14,color='#20252b',label='Observed regional mean')
    ax.set_xticks(xx[::3],[p['geos'][i] for i in order[::3]],rotation=45)
    ax.set(title='Prior heterogeneity versus observed regional averages',xlabel='Region (ordered by observed training average)',ylabel='Conversions per capita / training mean')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'prior_predictive_regions.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    lags=np.arange(LAG+1)
    for d in range(40):
        lam=r['decay'][d,0];w=lam**lags;w/=w.sum()
        axes[0].plot(lags,w,color='#4263b0',alpha=.15)
        a=np.linspace(0,4,200);axes[1].plot(a,a/(a+r['k'][d,0]),color='#b35431',alpha=.15)
    axes[0].set(title='40 prior adstock draws (Channel0)',xlabel='Lag (weeks)',ylabel='Normalized weight')
    axes[1].set(title='40 prior Hill curves (s = 1)',xlabel='Scaled adstocked exposure',ylabel='Fraction of response ceiling')
    fig.tight_layout();fig.savefig(out/'prior_response_curves.png');plt.close(fig)
    report={'seed':SEED,'draws':len(r['predicted']),'training_weeks':n,'warmup_weeks_excluded':LAG,
            'effective_training_weeks':n-LAG,'holdout_weeks':HOLDOUT,
            'train_end':p['dates'][n-1],'holdout_start':p['dates'][n],
            'outcome_per_capita_scale':float(p['yscale']),
            'media_per_capita_scales':p['media_scales'].tolist(),
            'control_centers':p['control_center'].tolist(),'control_sds':p['control_sd'].tolist(),
            'prior_predictive_negative_cell_fraction':r['negative_fraction'],
            'prior_predictive_cell_fraction_above_3x_training_mean_per_capita':r['above_three_fraction'],
            'mean_paid_component_scaled_quantiles_5_50_95':np.quantile(r['paid_component'],[.05,.5,.95]).tolist(),
            'status':'Prior simulation only. No posterior, no calibrated causal effects, no holdout evaluation.',
            'caveat':'Priors use training-outcome scaling; this is data-informed scale elicitation, not a fully data-independent prior check.'}
    report.update(extra_diagnostics(p,r,out))
    (out/'prior_predictive_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def extra_diagnostics(p,r,out):
    """Pool draws and training cells with equal region-week weight; do not clip."""
    mu=r['mu_cells']; yp=r['y_cells']; c=r['components']
    assert np.allclose(c[...,:4].sum(axis=-1),mu)
    assert np.allclose(c.sum(axis=-1),yp)
    obs=p['y_scaled'][:,LAG:p['cutoff']].ravel()
    negmu=mu<0; negy=yp<0; total=yp.size
    counts={
        'mu_negative_y_negative':int((negmu & negy).sum()),
        'mu_nonnegative_y_negative':int((~negmu & negy).sum()),
        'mu_negative_y_nonnegative':int((negmu & ~negy).sum()),
        'mu_nonnegative_y_nonnegative':int((~negmu & ~negy).sum())}
    assert sum(counts.values())==total
    assert (counts['mu_negative_y_negative']+counts['mu_nonnegative_y_negative'])/total==r['negative_fraction']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1.5,1]})
    bins=np.linspace(min(yp.min(),mu.min(),obs.min()),max(yp.max(),mu.max(),obs.max()),140)
    for values,label,color in [(yp.ravel(),'Prior predictive outcome Y*','#4263b0'),(mu.ravel(),'Prior expected outcome mu','#b35431'),(obs,'Observed training outcome','#20252b')]:
        density,edges=np.histogram(values,bins=bins,density=True)
        axes[0].stairs(density,edges,label=label,color=color,lw=1.5)
    axes[0].axvline(0,color='grey',ls='--',lw=1)
    axes[0].set(xlabel='Conversions per capita / training mean',ylabel='Density',title='Region-week distribution (full simulated range)')
    axes[0].legend(fontsize=8)
    rates=np.array([counts['mu_negative_y_negative'],counts['mu_nonnegative_y_negative']])/total*100
    axes[1].barh([1,0],rates,color=['#b35431','#4263b0'])
    axes[1].set_yticks([1,0],['Negative Y*, negative mu','Negative Y*, nonnegative mu'])
    for j,v in enumerate(rates):axes[1].text(v+.02,1-j,f'{v:.3f}%',va='center')
    axes[1].set(xlim=(0,max(rates)*1.35),xlabel='% of all draw x region-week observations',title='Two routes to negative predictions')
    fig.suptitle('Prior diagnostics: training cells only; no posterior fitted',fontsize=12)
    fig.tight_layout();fig.savefig(out/'prior_region_week_distribution.png');plt.close(fig)
    names=['Regional baseline','Paid media (all channels)','Controls (combined)','Trend + seasonality','Observation noise']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True,sharey=True)
    summaries={}
    for ax,label,values in [(axes[0],'All draw x region-week cells',c.reshape(-1,5)),(axes[1],'Only cells with negative expected outcome',c[negmu])]:
        if len(values)==0:continue
        q=np.quantile(values,[.05,.5,.95],axis=0);mean=values.mean(axis=0)
        summaries[label]={'q05':q[0].tolist(),'median':q[1].tolist(),'q95':q[2].tolist(),'mean':mean.tolist(),'cells':len(values)}
        for j in range(5):
            ax.plot([q[0,j],q[2,j]],[j,j],color='#4263b0',lw=3,alpha=.6)
            ax.scatter(q[1,j],j,color='#4263b0',s=25)
            ax.scatter(mean[j],j,color='#b35431',marker='x',s=35)
        ax.axvline(0,color='grey',ls='--',lw=1)
        ax.set(title=label,xlabel='Additive contribution (scaled outcome units)')
    axes[0].set_yticks(range(5),names);axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color='#4263b0',marker='o',label='Median and 5th-95th percentiles'),Line2D([0],[0],color='#b35431',marker='x',ls='',label='Mean')],loc='lower center',ncol=2,fontsize=8)
    fig.suptitle('Prior component distributions: conditional patterns are not causal attribution',fontsize=12)
    fig.tight_layout(rect=[0,.08,1,.96]);fig.savefig(out/'prior_component_decomposition.png');plt.close(fig)
    result={'diagnostic_total_draw_cells':total,'negative_expected_outcome_fraction':float(negmu.mean()),
            'negative_prediction_cross_counts':counts,
            'negative_predictions_with_nonnegative_mean_share':float((~negmu & negy).sum()/negy.sum()),
            'observed_training_negative_fraction':float((obs<0).mean()),
            'distribution_quantiles':{label:np.quantile(val,[0,.01,.05,.5,.95,.99,1]).tolist() for label,val in [('observed',obs),('prior_expected',mu),('prior_predictive',yp)]},
            'component_names':names,'component_summaries':summaries,
            'diagnostic_note':'Equal cell weighting; prior draws share parameters across cells, so cells are not independent trials. Conditional component shifts are descriptive, not causal blame. Quantiles of components do not add to quantiles of totals.'}
    return result


def main():
    p=prepare();r=simulate(p);report=make_report(p,r)
    print(json.dumps(report,indent=2));return p,r,report


if __name__=='__main__': main()
