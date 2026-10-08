from pathlib import Path
import sys,json,argparse,hashlib,platform,importlib.metadata
import numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_experiment as r
parser=argparse.ArgumentParser(description='GMM configurable-window sensor ablation and recording audit')
parser.add_argument('--window',type=int,default=3)
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'sensor_audit')
args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
samples,_,provenance=r.prepare(args.data_dir)
metadata={'files':[{k:v for k,v in f.items() if k!='path'}|{'filename':Path(f['path']).name} for f in provenance],
          'python':platform.python_version(),'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn']},
          'window':args.window,'stride':1,'seeds':[0,1,2],'split_clips':{'train':305,'validation':54,'calibration':120,'normal_test':120,'fault_test':21},
          'threshold':'normal calibration score 99th quantile, strict >','warmup':'False, included in all-timestamp evaluation'}

out={'metadata':metadata,'recording':{},'ablation':[]}
for src,g in samples.groupby('source'):
 c=g.AI2_Current.to_numpy();a=c/1.1920929
 out['recording'][src]={'rows':len(c),'unique':len(np.unique(c)),'grid_fraction_tol_0.01':float(np.mean(abs(a-np.round(a))<.01)),'grid_fraction_tol_1e-5':float(np.mean(abs(a-np.round(a))<1e-5))}
for sensors in [['AI0_Vibration','AI1_Vibration','AI2_Current'],['AI0_Vibration','AI1_Vibration'],['AI2_Current']]:
 r.SENSORS=sensors;t=samples.loc[samples.split=='train',sensors].to_numpy();mu=t.mean(0);sd=t.std(0)
 ds={}
 for split in ['train','calibration','normal_test','fault_test']:
  xs=[];ys=[];idx=[]
  for _,cl in samples.loc[samples.split.eq(split)].groupby('clip_id',sort=False):
   vals=(cl[sensors].to_numpy()-mu)/sd
   for i in range(args.window,len(cl)):
    xs.append(vals[i-args.window:i].ravel());ys.append(vals[i]);idx.append(cl.index[i])
  ds[split]=(np.asarray(xs),np.asarray(ys),np.asarray(idx))
 z={k:np.concatenate([v[0],v[1]],axis=1) for k,v in ds.items()}
 for seed in [0,1,2]:
  fits=[GaussianMixture(n_components=k,covariance_type='full',reg_covar=1e-4,n_init=2,max_iter=200,random_state=seed).fit(z['train']) for k in [1,2,4]]
  m=min(fits,key=lambda m:m.bic(z['train']));th=np.quantile(-m.score_samples(z['calibration']),.99)
  p=samples.loc[samples.split.isin(['normal_test','fault_test'])].copy();p['pred']=False
  for split in ['normal_test','fault_test']:p.loc[ds[split][2],'pred']=-m.score_samples(z[split])>th
  a=p.y_true.eq(1);q=p.pred;tp=int((a&q).sum());fp=int((~a&q).sum());fn=int((a&~q).sum());tn=int((~a&~q).sum())
  quiet={c:float(p.loc[p.clip_id.eq(c),'pred'].mean()) for c in ['fault_0004','fault_0020','fault_0021']}
  delay=p.loc[a].groupby('clip_id').apply(lambda g:g.loc[g.pred,'elapsed_seconds'].min(),include_groups=False)
  item={'sensors':sensors,'seed':seed,'K':m.n_components,'F1':2*tp/(2*tp+fp+fn),'FAR':fp/(fp+tn),'miss':fn/(tp+fn),'early_0.5':float(delay.le(.5).mean()),'quiet_clips':quiet,'TP':tp,'FP':fp,'TN':tn,'FN':fn,'threshold':float(th)}
  out['ablation'].append(item)
  group='all_sensors' if len(sensors)==3 else 'vibration_only' if len(sensors)==2 else 'current_only'
  p[['clip_id','sample_index','elapsed_seconds','y_true','pred']].to_csv(args.output_dir/f'{group}_seed{seed}_predictions.csv',index=False)
  if len(sensors)==3 and seed==0:
   quiet_mask=p.y_true.eq(1)&p.AI0_Vibration.abs().le(.35)&p.sample_index.ge(args.window)
   out['existing_quiet']={'count':int(quiet_mask.sum()),'alarm_rate':float(p.loc[quiet_mask,'pred'].mean())}
  if len(sensors)==2 and seed==0:
   import joblib
   joblib.dump(m,args.output_dir/'vibration_only_seed0.joblib')
  print(json.dumps(item),flush=True)

print(json.dumps(out['recording']));print(json.dumps(out['existing_quiet']))
(args.output_dir/'audit.json').write_text(json.dumps(out,indent=2))
runs=[]
for item in out['ablation']:
 item=item.copy();item['input']='all_sensors' if len(item['sensors'])==3 else 'vibration_only' if len(item['sensors'])==2 else 'current_only'
 for k,v in item.pop('quiet_clips').items():item[k+'_alarm_rate']=v
 item['sensors']=','.join(item['sensors']);runs.append(item)
d=pd.DataFrame(runs);d.to_csv(args.output_dir/'seed_runs.csv',index=False)
a=d.groupby('input')[['F1','FAR','miss','early_0.5']].agg(['mean','std']);a.columns=['_'.join(c) for c in a.columns];a.to_csv(args.output_dir/'summary.csv')
