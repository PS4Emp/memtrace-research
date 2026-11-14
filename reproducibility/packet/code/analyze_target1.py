#!/usr/bin/env python3
"""Only consumes actual completed run arrays. No simulated scientific results."""
from pathlib import Path
import argparse,json,itertools
import numpy as np,pandas as pd
from bridge_core import bridge_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);records=[]
 for root in a.runs:
  complete=json.load(open(root/'complete.json'));assert complete['status']=='FORMAL_COMPLETE';seed=complete['seed']
  for key in complete['units']:
   z=dict(np.load(root/key/'actions.npz',allow_pickle=False));base={c:z[c+'__canonical'] for c in ['far','middle','recent']}
   for anchor in ['far','middle','recent','common_shuffle','mismatched_shuffle']:
    clamped={c:z[c+'__u_'+c+'__k_'+anchor] if anchor in ['far','middle','recent'] else z[c+'__'+anchor] for c in base}
    for r in bridge_metrics(base,clamped):records.append(dict(episode_key=key,family=key.split('_ep')[0],seed=seed,anchor=anchor,**r))
   for c in base:
    n=np.mean(np.abs(z[c+'__native'][:16]-base[c][:16]));records.append(dict(episode_key=key,family=key.split('_ep')[0],seed=seed,anchor='numerical_native_vs_common_reduction',pair=c,absolute_reduction=n))
 df=pd.DataFrame(records);df.to_csv(a.out/'all_pair_seed_metrics.csv',index=False)
 # Pairs and seeds are repeated measurements. Scientific resampling unit = episode.
 ep=df.groupby(['episode_key','family','anchor'],as_index=False).absolute_reduction.mean();ep.to_csv(a.out/'episode_endpoints.csv',index=False);rng=np.random.default_rng(42644);summ=[]
 for anchor,g in ep.groupby('anchor'):
  boots=np.zeros(20000);n=0
  for fam,h in g.groupby('family'):
   x=h.absolute_reduction.to_numpy();boots+=x[rng.integers(0,len(x),(20000,len(x)))].sum(1);n+=len(x)
  summ.append(dict(anchor=anchor,episodes=n,mean_episode_reduction=float(g.absolute_reduction.mean()),median_episode_reduction=float(g.absolute_reduction.median()),stratified_episode_bootstrap95=np.quantile(boots/n,[.025,.975]).tolist()))
 (a.out/'summary.json').write_text(json.dumps({'status':'EXPLORATORY_ANALYSIS_OF_SUPPLIED_ACTUAL_RUNS','results':summ},indent=2))
if __name__=='__main__':main()
