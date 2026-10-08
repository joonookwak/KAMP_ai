"""Predefined causal improvements; normal-only fitting/calibration, exploratory test comparison."""
from pathlib import Path
import sys,json,argparse,time,shutil
import numpy as np,pandas as pd,joblib
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/improvements';OUT.mkdir(parents=True,exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--code-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.code_dir));import run_experiment as r
data,_,provenance=r.prepare(args.data_dir);S=r.SENSORS
protocol={'seeds':[0,1,2],'study':'exploratory; prior test error analysis informed candidate mechanisms',
 'thresholds':'normal calibration q99; conditional and joint variants independently recalibrated',
 'candidate_definitions':{
  'Original':'saved original GMM W2',
  'ConditionalCurrent':'-log p(current three points | two vibration three points), same saved GMM',
  'JointConditional':'min of full NLL and conditional current NLL standardized by normal validation median/IQR; calibration q99',
  'MoreComponents':'K=1,2,4,8,16 train BIC; original 305 normal training clips',
  'MoreNormalCoverage':'305 train +54 previously unused validation normal clips, K=1,2,4 BIC; calibration/test unchanged',
  'StartupMarginal':'saved original GMM marginal at i=0 current only, i=1 last2; q99 matching calibration start index; W2 thereafter',
  'StartupConditional':'conditional current marginal at i=0/1, index-specific calibration; conditional current W2 thereafter',
  'StartupJointConditional':'full/conditional standardized-min score at i=0/1 with corresponding normal calibration; joint W2 thereafter'},
 'alarm_policies':{'point':'instant score > threshold','consecutive2':'two successive exceedances; warmup resets',
  'consecutive3':'three successive exceedances; warmup resets',
  'hysteresis2':'trigger above normal q99, clear after 2 consecutive scores <= normal q95; reset at each clip'},
 'selection_rule':'retain 100% fault first alarm by 0.5s; mean FP <= original, normal alerted clips <=2 every seed; maximize mean timestamp F1; if no improvement retain original',
 'future_samples':0,'retrospective_point_adjustment':False,'raw_data_uploaded':False}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))

def marginal(model,z,cols):
    logs=[]
    for k,w in enumerate(model.weights_):
        cov=model.covariances_[k][np.ix_(cols,cols)];d=z[:,cols]-model.means_[k,cols]
        sign,ld=np.linalg.slogdet(cov);assert sign>0
        logs.append(np.log(w)-.5*(len(cols)*np.log(2*np.pi)+ld+np.einsum('ij,jk,ik->i',d,np.linalg.inv(cov),d)))
    return -logsumexp(np.stack(logs),axis=0)

def fit_gmm(z,ks,seed):
    fits=[GaussianMixture(n_components=k,covariance_type='full',reg_covar=1e-4,n_init=2,max_iter=200,random_state=seed).fit(z) for k in ks]
    bs=[float(m.bic(z)) for m in fits];ix=int(np.argmin(bs));return fits[ix],{'K':fits[ix].n_components,'BIC':dict(zip(ks,bs)),'converged':bool(fits[ix].converged_)}

def policy(df,kind):
    pred=np.zeros(len(df),dtype=bool)
    for _,g in df.groupby('clip_id',sort=False):
        count=clear=0;active=False
        for ix,row in g.iterrows():
            if not np.isfinite(row.score):count=clear=0;active=False;continue
            exceed=row.score>row.threshold
            if kind=='point':pred[ix]=exceed
            elif kind.startswith('consecutive'):
                count=count+1 if exceed else 0;pred[ix]=count>=int(kind[-1])
            elif kind=='hysteresis2':
                if exceed:active=True;clear=0
                elif active:
                    clear=clear+1 if row.score<=row.clear_threshold else 0
                    if clear>=2:active=False;clear=0
                pred[ix]=active
    return pred

def metrics(df,pred):
    y=df.y_true.to_numpy().astype(bool);tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    q=df.copy();q['pred']=pred
    delay=q[q.y_true.eq(1)].groupby('clip_id').apply(lambda g:g.loc[g.pred,'elapsed_seconds'].min(),include_groups=False)
    categories={'FN_warmup':0,'FN_before_first_alarm':0,'FN_after_first_alarm':0,'FN_no_alarm_clip':0}
    for clip,g in q[q.y_true.eq(1)].groupby('clip_id'):
        miss=g[~g.pred];warm=miss.score.isna();categories['FN_warmup']+=int(warm.sum());ms=miss[~warm];first=delay[clip]
        if np.isnan(first):categories['FN_no_alarm_clip']+=len(ms)
        else:
            categories['FN_before_first_alarm']+=int(ms.elapsed_seconds.lt(first).sum());categories['FN_after_first_alarm']+=int(ms.elapsed_seconds.ge(first).sum())
    return {'F1':2*tp/(2*tp+fp+fn),'FP':fp,'FN':fn,'TP':tp,'TN':tn,'FAR':fp/(fp+tn),'miss':fn/(tp+fn),
      'normal_alarm_clips':int(q[q.y_true.eq(0)].groupby('clip_id').pred.any().sum()),
      'no_alarm_fault_clips':int(delay.isna().sum()),'early_0.0':float(delay.le(0).mean()),'early_0.2':float(delay.le(.2+1e-9).mean()),
      'early_0.3':float(delay.le(.3+1e-9).mean()),'early_0.5':float(delay.le(.5+1e-9).mean()),
      'delay_mean':float(delay.mean()),'delay_p90':float(delay.quantile(.9)),**categories}

runs=[];details=[]
for seed in [0,1,2]:
    base=args.experiment_root/('window_sweep/natural/results' if seed==0 else f'seed_experiments/natural/seed_{seed}/results')
    cfg=json.loads((base/'experiment_config.json').read_text());mu=np.array(cfg['normalization_mean']);sd=np.array(cfg['normalization_scale']);model=joblib.load(base/'models/GMM_W2.joblib')
    assert [v['sha256'] for v in cfg['source_files']]==[v['sha256'] for v in provenance]
    test=data[data.split.isin(['normal_test','fault_test'])].reset_index(drop=True)
    caldata=data[data.split.eq('calibration')].reset_index(drop=True)
    ds={s:r.windows(data,s,2,mu,sd) for s in ['train','validation','calibration','normal_test','fault_test']}
    tz=np.concatenate([np.c_[ds[s][0],ds[s][1]] for s in ['normal_test','fault_test']])
    testrows=np.concatenate([ds[s][2] for s in ['normal_test','fault_test']]);lookup={ix:j for j,ix in enumerate(data.index[data.split.isin(['normal_test','fault_test'])])};testindex=np.array([lookup[ix] for ix in testrows])
    cz=np.c_[ds['calibration'][0],ds['calibration'][1]];vz=np.c_[ds['validation'][0],ds['validation'][1]]
    scores={};full_test=-model.score_samples(tz);full_cal=-model.score_samples(cz);full_val=-model.score_samples(vz)
    ct=full_test-marginal(model,tz,[0,1,3,4,6,7]);cc=full_cal-marginal(model,cz,[0,1,3,4,6,7]);cv=full_val-marginal(model,vz,[0,1,3,4,6,7])
    scores['Original']=(full_test,full_cal)
    scores['ConditionalCurrent']=(ct,cc)
    center=np.array([np.median(full_val),np.median(cv)]);scale=np.array([np.subtract(*np.quantile(full_val,[.75,.25])),np.subtract(*np.quantile(cv,[.75,.25]))]);assert (scale>0).all()
    joint=lambda x,y:np.min((np.c_[x,y]-center)/scale,axis=1)
    scores['JointConditional']=(joint(full_test,ct),joint(full_cal,cc))
    for name,trainids,ks in [('MoreComponents',['train'],[1,2,4,8,16]),('MoreNormalCoverage',['train','validation'],[1,2,4])]:
        norm=data[data.split.isin(trainids)][S].to_numpy();nm=norm.mean(0);ns=norm.std(0)
        zd={s:r.windows(data,s,2,nm,ns) for s in ['train','validation','calibration','normal_test','fault_test']}
        zfit=np.concatenate([np.c_[zd[s][0],zd[s][1]] for s in trainids]);m,info=fit_gmm(zfit,ks,seed)
        scores[name]=(-m.score_samples(np.concatenate([np.c_[zd[s][0],zd[s][1]] for s in ['normal_test','fault_test']])), -m.score_samples(np.c_[zd['calibration'][0],zd['calibration'][1]]))
        info.update({'method':name,'seed':seed});details.append(info);joblib.dump(m,OUT/f'{name}_seed{seed}.joblib')
    for name,(sc,cal) in scores.items():
        d=test[['clip_id','sample_index','elapsed_seconds','y_true']].copy();d['score']=np.nan;d.loc[testindex,'score']=sc;d['threshold']=np.quantile(cal,.99);d['clear_threshold']=np.quantile(cal,.95)
        # Original saved scores and decisions must exactly match the baseline.
        if name=='Original':
            saved=pd.read_csv(base/'predictions/GMM_W2.csv');assert np.allclose(saved.anomaly_score.dropna(),d.score.dropna(),rtol=1e-9,atol=1e-9);assert np.array_equal(d.score>d.threshold,saved.y_pred)
        variants=[(name,d)]
        if name in ['Original','ConditionalCurrent','JointConditional']:
            startup=d.copy()
            for index in [0,1]:
                cols=list(range(6 if index==0 else 3,9))
                # Means/covariance of saved 9D GMM marginalized to observed prefix.
                def start_scores(frame):
                    zz=[];ixs=[]
                    for _,g in frame.groupby('clip_id',sort=False):
                        if len(g)<=index:continue
                        a=(g[S].iloc[:index+1].to_numpy()-mu)/sd;z=np.zeros(9);z[cols]=a.ravel();zz.append(z);ixs.append(g.index[index])
                    z=np.array(zz);f=marginal(model,z,cols)
                    c=f-marginal(model,z,[j for j in cols if j%3!=2])
                    return np.array(ixs),f,c
                ci,cf,cc0=start_scores(caldata);ii,tf,tc=start_scores(test)
                if name=='Original':ss,cs=tf,cf
                elif name=='ConditionalCurrent':ss,cs=tc,cc0
                else:
                    valdata=data[data.split.eq('validation')].reset_index(drop=True);_,vf,vc=start_scores(valdata)
                    cen=np.array([np.median(vf),np.median(vc)]);sca=np.array([np.subtract(*np.quantile(vf,[.75,.25])),np.subtract(*np.quantile(vc,[.75,.25]))]);assert (sca>0).all()
                    ss=np.min((np.c_[tf,tc]-cen)/sca,axis=1);cs=np.min((np.c_[cf,cc0]-cen)/sca,axis=1)
                startup.loc[ii,'score']=ss;startup.loc[ii,'threshold']=np.quantile(cs,.99);startup.loc[ii,'clear_threshold']=np.quantile(cs,.95)
            variants.append(({'Original':'StartupMarginal','ConditionalCurrent':'StartupConditional','JointConditional':'StartupJointConditional'}[name],startup))
        for variant,d in variants:
            for pol in ['point','consecutive2','consecutive3','hysteresis2']:
                pred=policy(d,pol);result={'variant':variant,'policy':pol,'seed':seed,**metrics(d,pred)}
                runs.append(result);q=d.copy();q['pred']=pred;q.to_csv(OUT/f'{variant}_{pol}_seed{seed}_predictions.csv',index=False)
                print(json.dumps(result),flush=True)
    pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False)
(OUT/'model_fits.json').write_text(json.dumps(details,indent=2))
df=pd.DataFrame(runs);summary=df.groupby(['variant','policy']).agg({c:['mean','std'] for c in df.columns if c not in ['variant','policy','seed']});summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(OUT/'summary.csv')
baseline=df[(df.variant=='Original')&(df.policy=='point')].FP.mean()
eligible=[]
for (variant,pol),g in df.groupby(['variant','policy']):
    if g['early_0.5'].eq(1).all() and g.normal_alarm_clips.le(2).all() and g.FP.mean()<=baseline:
        eligible.append({'variant':variant,'policy':pol,'F1':g.F1.mean(),'FP':g.FP.mean(),'FN':g.FN.mean(),'delay_mean':g.delay_mean.mean()})
eligible=sorted(eligible,key=lambda a:(-a['F1'],a['FP'],a['delay_mean']))
(OUT/'selection.json').write_text(json.dumps({'rule':protocol['selection_rule'],'eligible':eligible,'recommended':eligible[0]},indent=2))
deploy=OUT/'recommended';deploy.mkdir(exist_ok=True)
for seed in [0,1,2]:
    base=args.experiment_root/('window_sweep/natural/results' if seed==0 else f'seed_experiments/natural/seed_{seed}/results')
    config=json.loads((base/'experiment_config.json').read_text())
    saved=pd.read_csv(OUT/f'StartupMarginal_point_seed{seed}_predictions.csv')
    thresholds=[float(saved.loc[saved.sample_index.eq(i),'threshold'].iloc[0]) for i in [0,1,2]]
    shutil.copy2(base/'models/GMM_W2.joblib',deploy/f'GMM_seed{seed}.joblib')
    (deploy/f'config_seed{seed}.json').write_text(json.dumps({'seed':seed,'sensor_order':config['sensor_order'],'mean':config['normalization_mean'],'scale':config['normalization_scale'],'threshold_1_2_3_observations':thresholds,'policy':'point','reset':'call reset at each clip boundary','model_file':f'GMM_seed{seed}.joblib'},indent=2))
print('COMPLETE',len(runs),'BEST',json.dumps(eligible[0]),flush=True)
