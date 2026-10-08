"""Causal, per-timestamp predictions; labels are never used by the detector."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.detector import StartupMarginalGMM
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/test.csv')
    p.add_argument('--models-dir',type=Path,default=ROOT/'models')
    p.add_argument('--output',type=Path,default=ROOT/'runs/test_predictions.csv')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--mode',choices=['startup','original'],default='startup')
    args=p.parse_args()
    model=StartupMarginalGMM(args.models_dir,args.seed)
    df=pd.read_csv(args.input);sensors=['AI0_Vibration','AI1_Vibration','AI2_Current']
    required=['clip_id','sample_index','elapsed_seconds',*sensors]
    if not set(required).issubset(df): raise ValueError(f'Required columns: {required}')
    if not np.isfinite(df[sensors].to_numpy()).all(): raise ValueError('Non-finite sensor values')
    if df[['clip_id','sample_index']].duplicated().any(): raise ValueError('Duplicate clip/sample key')
    rows=[]
    for cid,g in df.groupby('clip_id',sort=False):
        if g.sample_index.tolist()!=list(range(len(g))):
            raise ValueError(f'Clip {cid} must start at sample_index=0 and be in order')
        if not g.elapsed_seconds.is_monotonic_increasing: raise ValueError('Time order invalid')
        model.reset()
        for row in g.to_dict('records'):
            result=model.update([row[s] for s in sensors])
            if args.mode=='original' and result['observed_points']<model.n_points:
                result.update(anomaly=False,score=np.nan,threshold=float(model.thresholds[-1]))
            identity={k:row[k] for k in ['source','source_row','clip_id','sample_index','timestamp','elapsed_seconds','y_true'] if k in row}
            rows.append({**identity,'seed':args.seed,'mode':args.mode,'pred':result['anomaly'],
                         'score':result['score'],'threshold':result['threshold'],'dimension':result['dimension']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print(f'{len(rows)} timestamp predictions -> {args.output}')

if __name__=='__main__': main()
