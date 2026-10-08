"""Reproduce report window/sensor/recording sensitivity checks using normal-only GMM."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd,joblib
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset import prepare,SENSORS
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

def marginal_nll(model, values, columns):
    terms=[]
    for k,w in enumerate(model.weights_):
        cov=model.covariances_[k][np.ix_(columns,columns)]
        L=cholesky(cov,lower=True)
        delta=values-model.means_[k,columns]
        transformed=solve_triangular(L,delta.T,lower=True)
        terms.append(np.log(w)-.5*(len(columns)*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+(transformed**2).sum(0)))
    score=-logsumexp(np.stack(terms),axis=0)
    if not np.isfinite(score).all():raise ValueError("Non-finite marginal score")
    return score
from evaluate import evaluate

WIDTHS=[1,2,3,5,8,10,15,20,30]

def vectors(frame,sensors,mean,scale,width,condition):
    output={n:([],[]) for n in range(1,width+2)}
    for _,g in frame.groupby('clip_id',sort=False):
        values=(g[sensors].to_numpy()-mean)/scale
        for i in range(len(g)):
            n=min(i+1,width+1);a=values[i-n+1:i+1]
            if condition=='level_removed':
                v=np.c_[a[:,sensors.index('AI0_Vibration')],a[:,sensors.index('AI1_Vibration')]].ravel()
                a=np.r_[v,np.diff(a[:,sensors.index('AI2_Current')])]
            else:a=a.ravel()
            output[n][0].append(a);output[n][1].append(g.index[i])
    return {n:(np.asarray(vals).reshape(len(vals),-1) if len(vals) else np.empty((0,0)),np.asarray(ix,dtype=int)) for n,(vals,ix) in output.items()}

def columns(n,width,nsensors,condition):
    if condition=='level_removed':
        return list(range(2*(width+1-n),2*(width+1)))+list(range(2*(width+1)+(width+1-n),3*(width+1)-1))
    return list(range(nsensors*(width+1-n),nsensors*(width+1)))

def one(samples,width,seed,sensors,condition='original',saved=None):
    d=samples.copy()
    if condition=='normal_rounded':
        ix=d.source.eq('normal');d.loc[ix,'AI2_Current']=np.round(d.loc[ix,'AI2_Current']/1.1920929)*1.1920929
    elif condition=='fault_jittered':
        ix=d.source.eq('fault');d.loc[ix,'AI2_Current']+=np.random.default_rng(seed).uniform(-1.1920929/2,1.1920929/2,ix.sum())
    train=d[d.split.eq('train')];cal=d[d.split.eq('calibration')];test=d[d.split.isin(['normal_test','fault_test'])]
    vals=train[sensors].to_numpy();mean=vals.mean(0);scale=vals.std(0)
    datasets=[vectors(x,sensors,mean,scale,width,condition) for x in [train,cal,test]]
    full=datasets[0][width+1][0]
    if saved:
        config=json.loads((saved/'experiment_config.json').read_text())
        assert np.allclose(mean,config['normalization_mean'],rtol=1e-12,atol=1e-12)
        assert np.allclose(scale,config['normalization_scale'],rtol=1e-12,atol=1e-12)
        model=joblib.load(saved/f'models/GMM_W{width}.joblib')
    else:
        models=[GaussianMixture(n_components=k,covariance_type='full',reg_covar=1e-4,n_init=2,max_iter=200,random_state=seed).fit(full) for k in [1,2,4]]
        model=min(models,key=lambda m:m.bic(full))
    original_threshold=float(np.quantile(-model.score_samples(datasets[1][width+1][0]),.99))
    rows=test[['source','source_row','clip_id','sample_index','timestamp','elapsed_seconds','y_true']].copy()
    rows['seed']=seed;rows['mode']='startup';rows['pred']=False;rows['score']=np.nan;rows['threshold']=np.nan
    for n in range(1,width+2):
        z,ix=datasets[2][n];cz,cix=datasets[1][n]
        if not len(ix):continue
        cols=columns(n,width,len(sensors),condition)
        if n==width+1:score=-model.score_samples(z);threshold=original_threshold
        else:
            # Prefix thresholds use calibration clips at the same sample index only.
            keep=cal.loc[cix,'sample_index'].to_numpy()==n-1
            threshold=float(np.quantile(marginal_nll(model,cz[keep],cols),.99));score=marginal_nll(model,z,cols)
        rows.loc[ix,'score']=score;rows.loc[ix,'threshold']=threshold;rows.loc[ix,'pred']=score>threshold
    final,delays=evaluate(rows);rows['mode']='original';rows.loc[rows.sample_index.lt(width),'pred']=False
    original,_=evaluate(rows)
    group='all_sensors' if len(sensors)==3 else 'vibration_only' if len(sensors)==2 else 'current_only'
    return [{**original,'window':width,'input':group,'condition':condition,'K':model.n_components},
            {**final,'window':width,'input':group,'condition':condition,'K':model.n_components}]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw');p.add_argument('--output-dir',type=Path,default=ROOT/'runs/report_checks')
    p.add_argument('--selected-window',type=int,default=3)
    p.add_argument('--study',choices=['windows','sensors','recording','all'],default='all')
    p.add_argument('--windows',nargs='+',type=int,default=WIDTHS)
    p.add_argument('--experiment-root',type=Path,default=None,help='Optional saved original window sweep; otherwise refit normal GMM')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);samples,_,_=prepare(args.data_dir);rows=[]
    with threadpool_limits(limits=1):
        for seed in [0,1,2]:
            studies=[]
            if args.study in ['windows','all']:
                studies.extend((w,SENSORS,'original') for w in args.windows)
            if args.study in ['sensors','all']:
                studies.extend((args.selected_window,s,'original') for s in [SENSORS,SENSORS[:2],SENSORS[2:]])
            if args.study in ['recording','all']:
                studies.extend((args.selected_window,SENSORS,c) for c in ['original','normal_rounded','fault_jittered','level_removed'])
            for width,sensors,condition in studies:
                saved=None
                if args.experiment_root and sensors==SENSORS and condition=='original':
                    rel='window_sweep/natural/results' if seed==0 else f'seed_experiments/natural/seed_{seed}/results'
                    saved=args.experiment_root/rel
                new=one(samples,width,seed,sensors,condition,saved);rows.extend(new)
                print(seed,width,','.join(sensors),condition,new[-1]['F1'],flush=True)
                pd.DataFrame(rows).to_csv(args.output_dir/'runs.csv',index=False)
    df=pd.DataFrame(rows).drop_duplicates(['seed','mode','window','input','condition'])
    metrics=['F1','FP','FN','false_alarm_rate','miss_rate','normal_alarm_clips','detected_by_0.5s','delay_mean_s']
    summary=df.groupby(['mode','window','input','condition'])[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    df.to_csv(args.output_dir/'runs.csv',index=False);summary.to_csv(args.output_dir/'summary.csv')

if __name__=='__main__':main()
