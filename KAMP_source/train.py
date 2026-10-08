"""Fit GMM and thresholds using NORMAL training/calibration only."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from src.dataset import SENSORS
from src.gmm import observed_windows, marginal_nll
ROOT=Path(__file__).resolve().parent

def train_one(train, cal, output, seed, hashes, window=3):
    if not train.y_true.eq(0).all() or not cal.y_true.eq(0).all():
        raise ValueError('Training and calibration must contain normal labels only')
    if set(train.clip_id) & set(cal.clip_id):
        raise ValueError('Training/calibration clips overlap')
    values=train[SENSORS].to_numpy(); mean=values.mean(0); scale=values.std(0)
    if not np.isfinite(values).all() or (scale<=0).any():
        raise ValueError('Invalid training values/scale')
    z=observed_windows(train,SENSORS,mean,scale, n_points=window+1)
    fitted=[]
    for k in [1,2,4]:
        model=GaussianMixture(n_components=k,covariance_type='full',reg_covar=1e-4,
                              n_init=2,max_iter=200,random_state=seed).fit(z)
        if not model.converged_: raise RuntimeError(f'GMM K={k} did not converge')
        fitted.append((float(model.bic(z)), k, model))
    _, k, model=min(fitted,key=lambda x:x[0])
    full_cal=observed_windows(cal,SENSORS,mean,scale, n_points=window+1)
    thresholds=[]
    for n in range(1,window+1):
        prefix=[]
        for _, clip in cal.groupby('clip_id',sort=False):
            if len(clip)<n: continue
            prefix.append(((clip[SENSORS].iloc[:n].to_numpy()-mean)/scale).ravel())
        thresholds.append(float(np.quantile(marginal_nll(model,np.asarray(prefix),list(range(3*(window+1-n),3*(window+1)))),.99)))
    thresholds.append(float(np.quantile(-model.score_samples(full_cal),.99)))
    output.mkdir(parents=True,exist_ok=True)
    filename=f'GMM_seed{seed}.joblib'; joblib.dump(model,output/filename)
    config={'seed':seed,'sensor_order':SENSORS,'mean':mean.tolist(),'scale':scale.tolist(),
            'window':window,'thresholds_by_observed_points':thresholds,'time_unit':'CSV observations, 0.1 seconds inside clip',
            'policy':'point','reset':'call reset() at clip boundary','model_file':filename}
    (output/f'config_seed{seed}.json').write_text(json.dumps(config,indent=2)+'\n')
    metadata={'seed':seed,'window':window,'full_dimension':3*(window+1),'training_windows':len(z),'chosen_components':k,'K_candidates_BIC':{str(n):b for b,n,_ in fitted},
              'normal_train_clips':int(train.clip_id.nunique()),'normal_calibration_clips':int(cal.clip_id.nunique()),
              'threshold_quantile':.99,'threshold_comparison':'strict >','input_SHA256':hashes,
              'test_data_used':False,'normalization':'train-only population standard deviation (ddof=0)'}
    (output/f'training_seed{seed}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'seed={seed} K={k} thresholds={thresholds}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/models')
    p.add_argument('--window',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=p.parse_args()
    train_path=args.data_dir/'train.csv'; cal_path=args.data_dir/'calibration.csv'
    train=pd.read_csv(train_path);cal=pd.read_csv(cal_path)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path,cal_path]}
    with threadpool_limits(limits=1):
        for seed in args.seeds: train_one(train,cal,args.output_dir,seed,hashes,args.window)

if __name__=='__main__': main()
