"""Eight baseline methods under original/grid/jitter/current-difference inputs."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import argparse,sys
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'comparisons/baselines'))
from src.dataset import prepare,SENSORS
from report_checks import vectors
from models import MODELS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--window',type=int,default=3);p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--output-dir',type=Path,default=ROOT/'runs/recording_baselines');args=p.parse_args()
 args.output_dir.mkdir(parents=True,exist_ok=True);samples,_,_=prepare(args.data_dir);rows=[]
 with threadpool_limits(limits=1):
  for condition in ['original','normal_rounded','fault_jittered','level_removed']:
   for name in ['GMM','Mahalanobis','LOF20','PCA_Recon','KDE','kNN20','OCSVM','IsolationForest']:
    for seed in ([0,1,2] if name in ['GMM','IsolationForest'] else [0]):
     d=samples.copy()
     if condition=='normal_rounded':
      ix=d.source.eq('normal');d.loc[ix,'AI2_Current']=np.round(d.loc[ix,'AI2_Current']/1.1920929)*1.1920929
     elif condition=='fault_jittered':
      ix=d.source.eq('fault');d.loc[ix,'AI2_Current']+=np.random.default_rng(seed).uniform(-1.1920929/2,1.1920929/2,ix.sum())
     train=d[d.split.eq('train')];vals=train[SENSORS].to_numpy();mean=vals.mean(0);scale=vals.std(0)
     windows={s:vectors(d[d.split.eq(s)],SENSORS,mean,scale,args.window,condition)[args.window+1] for s in ['train','validation','calibration','normal_test','fault_test']}
     model=MODELS[name](seed=seed)
     z=windows['train'][0];v=windows['validation'][0];model.fit(z[:,:-3],z[:,-3:],v[:,:-3],v[:,-3:])
     cal=windows['calibration'][0];th=float(np.quantile(model.score(cal[:,:-3],cal[:,-3:]),.99))
     test=d[d.split.isin(['normal_test','fault_test'])].copy();test['pred']=False
     for split in ['normal_test','fault_test']:
      z,ix=windows[split];test.loc[ix,'pred']=model.score(z[:,:-3],z[:,-3:])>th
     y=test.y_true.to_numpy().astype(bool);q=test.pred.to_numpy();tp=int((y&q).sum());fp=int((~y&q).sum());fn=int((y&~q).sum());tn=int((~y&~q).sum())
     row={'condition':condition,'method':name,'seed':seed,'F1':2*tp/(2*tp+fp+fn),'TP':tp,'FP':fp,'TN':tn,'FN':fn};rows.append(row);print(condition,name,seed,row['F1'],flush=True)
     pd.DataFrame(rows).to_csv(args.output_dir/'runs.csv',index=False)
 df=pd.DataFrame(rows);summary=df.groupby(['condition','method'])[['F1','FP','FN']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(args.output_dir/'summary.csv')
if __name__=='__main__':main()
