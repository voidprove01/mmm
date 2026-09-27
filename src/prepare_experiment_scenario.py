"""Build hypothetical intervention inputs directly from training data.

This does not depend on a generated experiment, its outcomes, or power studies.
"""
import json
import numpy as np
import pandas as pd
from prior_predictive import ROOT,prepare,adstock

p=prepare();idx=np.arange(p['cutoff']-48,p['cutoff']-4)
start,stop=24,32
x0=p['x'][:,idx].copy();x1=x0.copy();x1[:,start:stop,2]*=1.2
raw=pd.read_csv(ROOT/'data/geo_all_channels.csv').set_index(['geo','time']).reindex(pd.MultiIndex.from_product([p['geos'],p['dates']]))
rpc=raw.revenue_per_conversion.to_numpy().reshape(p['y'].shape)[:,idx]
spend=raw.Channel2_spend.to_numpy().reshape(p['y'].shape)[:,idx]
D=float(.2*spend[:,start:stop].sum());mask=np.arange(len(idx))>=start
out=ROOT/'data/illustrative_experiment';out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'inputs.npz',x_control=x0,x_treatment=x1,population=p['pop'][:,idx],yscale=p['yscale'],revenue_per_conversion=rpc,evaluation_mask=mask)
(out/'scenario.json').write_text(json.dumps(dict(full_population_incremental_spend=D,channel='Channel2',increase=.2,intervention_weeks=8,followup_weeks=12,geos=p['geos'],source_template_dates=p['dates'][idx].tolist(),description='Hypothetical independent experiment; input schedules only, no outcome data'),indent=2))
with np.load(ROOT/'reports/mcmc/posterior_samples.npz') as file:s={k:file[k] for k in ['beta','decay','k']}
result=np.empty(s['decay'].shape[:2]);mult=p['pop'][:,idx]*p['yscale']*rpc
for c in range(result.shape[0]):
 for d in range(result.shape[1]):
  a0=adstock(x0,s['decay'][c,d]);a1=adstock(x1,s['decay'][c,d]);k=s['k'][c,d]
  difference=(s['beta'][c,d,:,None,:]*(a1/(a1+k)-a0/(a0+k))).sum(axis=2)
  result[c,d]=(difference[:,mask]*mult[:,mask]).sum()
report=ROOT/'reports/illustrative_calibration';report.mkdir(parents=True,exist_ok=True)
np.savez_compressed(report/'uncalibrated_experiment_predictions.npz',revenue_lift=result)
print('Prepared hypothetical experiment inputs and original-posterior contrast. No simulated experimental outcome was used.')
