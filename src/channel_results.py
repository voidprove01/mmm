"""Finite-window channel interventions using the frozen hierarchical posterior."""
from pathlib import Path
import hashlib
import argparse
import json
import numpy as np
import pandas as pd
import arviz as az
from prior_predictive import ROOT, LAG, prepare, adstock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--posterior-report',default='mcmc')
    parser.add_argument('--output-report',default='channel_results')
    options=parser.parse_args()
    out = ROOT / 'reports' / options.output_report
    out.mkdir(parents=True, exist_ok=True)
    p = prepare()
    n = p['cutoff']
    sl = slice(LAG, n)
    raw = pd.read_csv(ROOT / 'data/geo_all_channels.csv').set_index(['geo', 'time'])
    raw = raw.reindex(pd.MultiIndex.from_product([p['geos'], p['dates']]))
    g, t, m = p['x'].shape
    spend_cells = raw[[f'Channel{i}_spend' for i in range(m)]].to_numpy().reshape(g,t,m)
    revenue_unit = raw.revenue_per_conversion.to_numpy().reshape(g,t)[:,sl]
    assert np.isfinite(spend_cells).all() and (spend_cells >= 0).all()
    assert np.isfinite(revenue_unit).all() and (revenue_unit >= 0).all()
    assert np.array_equal(spend_cells > 0, p['x'] > 0), 'Spend/exposure support mismatch'
    spend = spend_cells[:,sl].sum(axis=(0,1))
    multiplier = p['pop'][:,sl] * p['yscale']
    revenue_multiplier = multiplier * revenue_unit
    sample_path = ROOT / 'reports' / options.posterior_report / 'posterior_samples.npz'
    with np.load(sample_path) as archive:
        samples = {name: archive[name] for name in ['decay','k','beta','alpha','delta','gamma']}
    chains, draws = samples['decay'].shape[:2]
    effects = np.empty((chains,draws,m))
    revenue = np.empty_like(effects)
    marginal = np.empty_like(effects)
    x = p['x'][:,:n].copy()
    x_pre = x.copy()
    x_pre[:,LAG:] = 0
    finite_difference_errors = []
    for c in range(chains):
        for d in range(draws):
            decay, k, beta = (samples[name][c,d] for name in ['decay','k','beta'])
            a = adstock(x,decay)[:,sl]
            pre = adstock(x_pre,decay)[:,sl]
            window = a-pre
            h = a/(a+k)
            h0 = pre/(pre+k)
            effect = beta[:,None,:]*(h-h0)
            deriv = beta[:,None,:]*k*window/(a+k)**2
            effects[c,d] = (effect*multiplier[:,:,None]).sum(axis=(0,1))
            revenue[c,d] = (effect*revenue_multiplier[:,:,None]).sum(axis=(0,1))
            marginal[c,d] = (deriv*revenue_multiplier[:,:,None]).sum(axis=(0,1))/spend
            if d == 0:
                eps = 1e-5
                ap, am = a+eps*window, a-eps*window
                fd = beta[:,None,:]*(ap/(ap+k)-am/(am+k))/(2*eps)
                fd_roas = (fd*revenue_multiplier[:,:,None]).sum(axis=(0,1))/spend
                np.testing.assert_allclose(fd_roas,marginal[c,d],rtol=1e-7,atol=1e-8)
                finite_difference_errors.append(float(np.max(np.abs(fd_roas-marginal[c,d]))))
                # Full mean comparison verifies controls and other channels cancel.
                base = samples['alpha'][c,d,:,None]+p['basis'][sl]@samples['delta'][c,d]+p['z'][:,sl]@samples['gamma'][c,d]
                mu = base+(beta[:,None,:]*h).sum(axis=2)
                for j in range(m):
                    h_cf = h.copy(); h_cf[:,:,j] = h0[:,:,j]
                    mu_cf = base+(beta[:,None,:]*h_cf).sum(axis=2)
                    np.testing.assert_allclose(((mu-mu_cf)*multiplier).sum(),effects[c,d,j],rtol=1e-10)
    roas = revenue/spend
    assert (effects >= 0).all() and (marginal >= 0).all()
    assert (marginal <= roas+1e-9).all()  # Concavity under the chosen intervention.
    observed_revenue = float((p['y'][:,sl]*revenue_unit).sum())
    metrics = dict(incremental_conversions=effects,incremental_revenue=revenue,
                   revenue_share_pct=100*revenue/observed_revenue,roas=roas,marginal_roas=marginal)
    np.savez_compressed(out/'posterior_metrics.npz',**metrics)
    rows=[]
    for j in range(m):
        row=dict(channel=f'Channel{j}',spend=spend[j])
        for name, values in metrics.items():
            v=values[:,:,j].ravel()
            row[name+'_mean']=float(v.mean())
            row[name+'_lo90'],row[name+'_hi90']=np.quantile(v,[.05,.95])
        rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(out/'channel_results.csv',index=False)
    diag=az.summary(az.from_dict(posterior=metrics),kind='diagnostics',round_to='none')
    diag.to_csv(out/'derived_diagnostics.csv')
    meta=dict(start=str(p['dates'][LAG]),end=str(p['dates'][n-1]),weeks=n-LAG,regions=g,
              posterior_draws=chains*draws,observed_revenue=observed_revenue,
              interval='90% equal-tailed posterior credible interval; central estimate is posterior mean',
              intervention='Remove only in-window exposures; preserve earlier history; evaluate only in-window outcomes.',
              marginal='Derivative at historical spend, scaling each channel geo-week proportionally at fixed cell-level CPM.',
              currency='Unspecified source-data currency unit, assumed common across spend and revenue.',
              max_rhat=float(diag.r_hat.max()),min_ess_bulk=float(diag.ess_bulk.min()),
              max_derivative_check_error=max(finite_difference_errors),
              checks=['Spend/exposure positive support matches','Analytic derivative matches central finite difference in each chain',
                      'Full mean counterfactual matches component effect in each chain', 'Nonnegative effects and marginal ROAS <= average ROAS'],
              sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [sample_path,ROOT/'data/geo_all_channels.csv',Path(__file__)]})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    def fmt(row,name,scale=1):
        return f"{row[name+'_mean']/scale:.2f} [{row[name+'_lo90']/scale:.2f}, {row[name+'_hi90']/scale:.2f}]"
    lines=['| Channel | Spend (million CU) | Incremental conversions (million) | Incremental revenue (million CU) | Revenue share (%) | ROAS | Marginal ROAS |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+ ' | '.join([row['channel'],f"{row['spend']/1e6:.2f}",fmt(row,'incremental_conversions',1e6),fmt(row,'incremental_revenue',1e6),fmt(row,'revenue_share_pct'),fmt(row,'roas'),fmt(row,'marginal_roas')])+' |')
    (out/'table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(meta,indent=2)); print('\n'.join(lines))


if __name__ == '__main__':
    main()
