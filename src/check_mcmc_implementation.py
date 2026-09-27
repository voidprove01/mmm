"""Checks model equivalence and outcome-holdout isolation before inference."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from prior_predictive import prepare,adstock,ROOT
from fit_mcmc import inputs,verify

def main():
    p=prepare();verify(p,inputs(p))
    assert np.allclose(adstock(np.ones((2,30,5)),np.array([.1,.3,.5,.8,.95]))[:,12:],1)
    df=pd.read_csv(ROOT/'data/geo_all_channels.csv')
    df.loc[df.time>=p['dates'][p['cutoff']],'conversions']*=1000
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'altered.csv';df.to_csv(path,index=False);q=prepare(path)
    for original,changed in zip(inputs(p),inputs(q)):
        assert np.array_equal(np.asarray(original),np.asarray(changed))
    print('PASS: all MCMC inputs unchanged after perturbing holdout outcomes; normalized constant exposure preserved.')

if __name__=='__main__':main()
