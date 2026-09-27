"""High-information hypothetical experiment -> calibrated posterior.

The experiment summary is an explicitly hypothetical calibration input, NOT an estimate
from simulated_experiment_v1. Historical priors and likelihood are unchanged.
"""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'.matplotlib'))
import json,time,hashlib
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC,NUTS,init_to_median
import arviz as az
from fit_mcmc import model as historical_model,inputs,verify
from prior_predictive import ROOT,LAG,prepare,adstock

OUT=ROOT/'reports/illustrative_calibration'
EVIDENCE=ROOT/'data/illustrative_experiment'
SEED=2026092801


def calibrated_model(lagged,z,basis,y,experiment):
    with handlers.trace() as sites:
        historical_model(lagged,z,basis,y)
    beta=sites['beta']['value'][:,2]
    decay=sites['decay']['value'][2]
    k=sites['k']['value'][2]
    w=decay**jnp.arange(LAG+1);w=w/w.sum()
    a0=jnp.sum(experiment['lagged_control']*w,axis=2)
    a1=jnp.sum(experiment['lagged_treatment']*w,axis=2)
    lift=jnp.sum(beta[:,None]*(a1/(a1+k)-a0/(a0+k))*experiment['revenue_multiplier'])
    numpyro.deterministic('experiment_lift',lift)
    ret=numpyro.deterministic('experiment_return',lift/experiment['spend'])
    # Scaling the likelihood into return units changes only a theta-independent constant.
    numpyro.sample('illustrative_experiment',dist.Normal(ret,experiment['se_return']),obs=experiment['observed_return'])


def main():
    numpyro.set_host_device_count(4)
    OUT.mkdir(parents=True,exist_ok=True);EVIDENCE.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT/'data/illustrative_experiment/inputs.npz') as f:
        inp={k:f[k] for k in ['x_control','x_treatment','population','yscale','revenue_per_conversion','evaluation_mask']}
    scenario=json.loads((ROOT/'data/illustrative_experiment/scenario.json').read_text())
    spend=scenario['full_population_incremental_spend']
    evidence=dict(label='Hypothetical high-information experiment',
        observed_incremental_return=1.20,standard_error_incremental_return=.15,
        observed_incremental_revenue=spend*1.20,standard_error_incremental_revenue=spend*.15,
        full_population_incremental_spend=spend,
        estimand='Channel2 +20% for eight weeks, revenue over intervention plus twelve followup weeks, all 40 geos',
        provenance='Hypothetical summary estimate and standard error specified for the calibration scenario. Not derived from v1; no real experiment or achieved field precision is claimed.',
        rationale='Evaluate how independent evidence below the original model expectation changes posterior estimates and allocation decisions.',
        likelihood='Normal on experimental return; equivalent to Normal revenue likelihood up to a parameter-independent unit conversion.',
        independence='Hypothetical independent experiment; its summary is used exactly once; v1 outcome data are not added.',
        seed=SEED)
    (EVIDENCE/'evidence.json').write_text(json.dumps(evidence,indent=2))
    mask=inp['evaluation_mask'].astype(bool);times=np.flatnonzero(mask)
    exp={f'lagged_{policy}':jnp.asarray(np.stack([inp[f'x_{policy}'][:,times-j,2] for j in range(LAG+1)],axis=2)) for policy in ['control','treatment']}
    exp.update(revenue_multiplier=jnp.asarray((inp['population']*inp['yscale']*inp['revenue_per_conversion'])[:,mask]),
               spend=jnp.asarray(spend),observed_return=jnp.asarray(1.20),se_return=jnp.asarray(.15))
    p=prepare();args=inputs(p);verify(p,args)
    trace=handlers.trace(handlers.seed(calibrated_model,jax.random.PRNGKey(91))).get_trace(*args,exp)
    beta=np.asarray(trace['beta']['value']);decay=np.asarray(trace['decay']['value']);k=np.asarray(trace['k']['value'])
    a0=adstock(inp['x_control'],decay);a1=adstock(inp['x_treatment'],decay)
    expected=((beta[:,None,:]*(a1/(a1+k)-a0/(a0+k))).sum(2)[:,mask]*np.asarray(exp['revenue_multiplier'])).sum()
    np.testing.assert_allclose(expected,float(trace['experiment_lift']['value']),rtol=1e-10)
    r=expected/spend
    lp=-.5*((1.2-r)/.15)**2-np.log(.15)-.5*np.log(2*np.pi)
    np.testing.assert_allclose(lp,float(trace['illustrative_experiment']['fn'].log_prob(1.2)),rtol=1e-10)
    protected=[ROOT/'reports/mcmc/posterior_samples.npz',ROOT/'data/geo_all_channels.csv',ROOT/'data/illustrative_experiment/scenario.json',ROOT/'reports/budget_optimization/summary.json']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    (OUT/'run_protocol.json').write_text(json.dumps(dict(seed=SEED,chains=4,warmup=1000,draws=1000,target_accept=.99,
        original_hashes=hashes,checks=['Historical mean and likelihood match','Experimental contrast matches independent NumPy calculation','Experimental likelihood matches manual calculation'],
        evidence=evidence,priors_changed=False,holdout_used=False),indent=2))
    mass=[('alpha0','tau_alpha','z_alpha','b','tau','decay','k','gamma','delta','sigma')]
    kernel=NUTS(calibrated_model,target_accept_prob=.99,max_tree_depth=10,dense_mass=mass,init_strategy=init_to_median(num_samples=20))
    runner=MCMC(kernel,num_warmup=1000,num_samples=1000,num_chains=4,chain_method='parallel',progress_bar=False)
    print('Likelihood checks passed. Starting 4-chain calibrated NUTS.',flush=True)
    start=time.time();runner.run(jax.random.PRNGKey(SEED),*args,exp,extra_fields=('potential_energy','energy','num_steps','accept_prob'))
    jax.block_until_ready(runner.get_samples());elapsed=time.time()-start
    samples={k:np.asarray(v) for k,v in runner.get_samples(group_by_chain=True).items()}
    stats={k:np.asarray(v) for k,v in runner.get_extra_fields(group_by_chain=True).items()}
    np.savez_compressed(OUT/'posterior_samples.npz',**samples);np.savez_compressed(OUT/'sampler_stats.npz',**stats)
    idata=az.from_numpyro(runner,coords={'geo':p['geos'],'channel':[f'Channel{i}' for i in range(5)]},dims={'alpha':['geo'],'z_alpha':['geo'],'beta':['geo','channel'],'z_beta':['geo','channel'],'b':['channel'],'tau':['channel'],'decay':['channel'],'k':['channel']},log_likelihood=False)
    idata.to_netcdf(OUT/'posterior.nc');diag=az.summary(idata,round_to='none');diag.to_csv(OUT/'parameter_summary.csv')
    report=dict(chains=4,warmup_per_chain=1000,draws_per_chain=1000,elapsed_seconds=elapsed,
        divergences=int(stats['diverging'].sum()),max_rhat=float(diag.r_hat.max()),min_bulk_ess=float(diag.ess_bulk.min()),min_tail_ess=float(diag.ess_tail.min()),
        max_mcse_over_sd=float((diag.mcse_mean/diag.sd).max()),tree_depth_limit_count=int((stats['num_steps']>=1023).sum()),bfmi_per_chain=np.asarray(az.bfmi(idata)).tolist(),
        priors_changed=False,calibration='Hypothetical Normal experimental likelihood',holdout_used=False,
        original_files_unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items()))
    (OUT/'diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
