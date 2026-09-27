"""NUTS for the documented hierarchical MMM. Priors unchanged."""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'.matplotlib'))
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC,NUTS,init_to_median
from numpyro import handlers
import arviz as az
from prior_predictive import prepare,adstock,ROOT,LAG


def model(lagged,z,basis,y=None):
    g,t,_,m=lagged.shape
    alpha0=numpyro.sample('alpha0',dist.Normal(.6,.2))
    tau_alpha=numpyro.sample('tau_alpha',dist.HalfNormal(.2))
    za=numpyro.sample('z_alpha',dist.Normal(0,1).expand([g]).to_event(1))
    alpha=numpyro.deterministic('alpha',alpha0+tau_alpha*za)
    b=numpyro.sample('b',dist.Normal(np.log(.15),.5).expand([m]).to_event(1))
    tau=numpyro.sample('tau',dist.HalfNormal(.3).expand([m]).to_event(1))
    zb=numpyro.sample('z_beta',dist.Normal(0,1).expand([g,m]).to_event(2))
    beta=numpyro.deterministic('beta',jnp.exp(b+tau*zb))
    decay=numpyro.sample('decay',dist.Beta(2,2).expand([m]).to_event(1))
    k=numpyro.sample('k',dist.LogNormal(0,.6).expand([m]).to_event(1))
    gamma=numpyro.sample('gamma',dist.Normal(0,.1).expand([4]).to_event(1))
    delta=numpyro.sample('delta',dist.Normal(jnp.zeros(3),jnp.array([.2,.1,.1])).to_event(1))
    sigma=numpyro.sample('sigma',dist.HalfNormal(.15))
    w=decay[None,:]**jnp.arange(LAG+1)[:,None]
    w=w/w.sum(axis=0)
    a=jnp.sum(lagged*w[None,None,:,:],axis=2)
    mu=alpha[:,None]+basis@delta+z@gamma+jnp.sum(beta[:,None,:]*a/(a+k),axis=-1)
    numpyro.sample('outcome',dist.Normal(mu,sigma).to_event(2),obs=y)


def inputs(p):
    n=p['cutoff']
    lagged=np.stack([p['x'][:,LAG-j:n-j,:] for j in range(LAG+1)],axis=2)
    return tuple(jnp.asarray(v) for v in [lagged,p['z'][:,LAG:n],p['basis'][LAG:n],p['y_scaled'][:,LAG:n]])


def verify(p,args):
    tr=handlers.trace(handlers.seed(model,jax.random.PRNGKey(12))).get_trace(*args)
    get=lambda k:np.asarray(tr[k]['value'])
    n=p['cutoff']; a=adstock(p['x'][:,:n],get('decay'))[:,LAG:]
    mu=get('alpha')[:,None]+p['basis'][LAG:n]@get('delta')+p['z'][:,LAG:n]@get('gamma')+(get('beta')[:,None,:]*a/(a+get('k'))).sum(axis=2)
    assert np.allclose(mu,np.asarray(tr['outcome']['fn'].base_dist.loc),rtol=1e-10,atol=1e-10)
    y=p['y_scaled'][:,LAG:n];s=get('sigma')
    manual=(-.5*((y-mu)/s)**2-np.log(s)-.5*np.log(2*np.pi)).sum()
    actual=float(tr['outcome']['fn'].log_prob(args[-1]))
    assert np.allclose(manual,actual,rtol=1e-10)
    print('PASS: model mean and full-data Gaussian log-likelihood match the prior simulator.',flush=True)


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--pilot',action='store_true');cli.add_argument('--draws',type=int,default=1000);cli.add_argument('--warmup',type=int,default=1000);cli.add_argument('--target',type=float,default=.9);cli.add_argument('--name',default='mcmc');cli.add_argument('--dense-hyper',action='store_true')
    opts=cli.parse_args();chains=1 if opts.pilot else 4
    numpyro.set_host_device_count(chains)
    p=prepare();args=inputs(p);verify(p,args)
    out=ROOT/'reports'/('pilot' if opts.pilot else opts.name);out.mkdir(parents=True,exist_ok=True)
    warmup=100 if opts.pilot else opts.warmup;draws=100 if opts.pilot else opts.draws
    mass=[('alpha0','tau_alpha','z_alpha','b','tau','decay','k','gamma','delta','sigma')] if opts.dense_hyper else False
    kernel=NUTS(model,target_accept_prob=opts.target,max_tree_depth=10,dense_mass=mass,init_strategy=init_to_median(num_samples=20))
    runner=MCMC(kernel,num_warmup=warmup,num_samples=draws,num_chains=chains,chain_method='parallel',progress_bar=True)
    start=time.time()
    runner.run(jax.random.PRNGKey(20260926 if opts.pilot else 20260925),*args,extra_fields=('potential_energy','energy','num_steps','accept_prob'))
    jax.block_until_ready(runner.get_samples())
    elapsed=time.time()-start
    samples={k:np.asarray(v) for k,v in runner.get_samples(group_by_chain=True).items()}
    extra={k:np.asarray(v) for k,v in runner.get_extra_fields(group_by_chain=True).items()}
    np.savez_compressed(out/'posterior_samples.npz',**samples)
    np.savez_compressed(out/'sampler_stats.npz',**extra)
    idata=az.from_numpyro(runner,coords={'geo':p['geos'],'channel':[f'Channel{i}' for i in range(5)]},dims={'alpha':['geo'],'z_alpha':['geo'],'beta':['geo','channel'],'z_beta':['geo','channel'],'b':['channel'],'tau':['channel'],'decay':['channel'],'k':['channel']},log_likelihood=False)
    idata.to_netcdf(out/'posterior.nc')
    diag=az.summary(idata,round_to=6);diag.to_csv(out/'parameter_summary.csv')
    report={'chains':chains,'warmup_per_chain':warmup,'draws_per_chain':draws,'elapsed_seconds':elapsed,'target_accept':opts.target,'dense_hyper':opts.dense_hyper,'divergences':int(extra['diverging'].sum()),'max_rhat':float(diag.r_hat.max()),'min_bulk_ess':float(diag.ess_bulk.min()),'min_tail_ess':float(diag.ess_tail.min()),'max_mcse_over_sd':float((diag.mcse_mean/diag.sd).max()),'rhat_ge_1_01_count':int((diag.r_hat>=1.01).sum()),'tree_depth_limit_count':int((extra['num_steps']>=1023).sum()),'bfmi_per_chain':np.asarray(az.bfmi(idata)).tolist(),'training_observations':int(args[-1].size),'holdout_used':False,'priors_changed':False,'parameterization':'noncentered alpha and log beta; same priors as prior predictive model','versions':{'numpyro':numpyro.__version__,'jax':jax.__version__,'arviz':az.__version__}}
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
