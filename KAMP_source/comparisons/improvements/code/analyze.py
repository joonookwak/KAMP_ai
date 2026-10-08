"""Read-only diagnostic of saved original 3-sensor GMM W2, seeds 0/1/2."""
from pathlib import Path
import argparse, sys, json, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
import joblib

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--experiment-root',type=Path,required=True)
parser.add_argument('--code-dir',type=Path,required=True)
parser.add_argument('--data-dir',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.code_dir));import run_experiment as r
S=r.SENSORS
data,segments,provenance=r.prepare(args.data_dir)
OUT=ROOT/'results';FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def marginal_nll(model,z,cols):
    logs=[]
    for k,w in enumerate(model.weights_):
        cov=model.covariances_[k][np.ix_(cols,cols)];d=z[:,cols]-model.means_[k,cols]
        sign,ld=np.linalg.slogdet(cov);assert sign>0
        dist=np.einsum('ij,jk,ik->i',d,np.linalg.inv(cov),d)
        logs.append(np.log(w)-.5*(len(cols)*np.log(2*np.pi)+ld+dist))
    return -logsumexp(np.stack(logs),axis=0)

seed_metrics=[];tables=[];fits={}
for seed in [0,1,2]:
    base=args.experiment_root/('window_sweep/natural/results' if seed==0 else f'seed_experiments/natural/seed_{seed}/results')
    cfg=json.loads((base/'experiment_config.json').read_text())
    assert [v['sha256'] for v in provenance]==[v['sha256'] for v in cfg['source_files']]
    mu=np.array(cfg['normalization_mean']);sd=np.array(cfg['normalization_scale'])
    model=joblib.load(base/'models/GMM_W2.joblib')
    d=pd.read_csv(base/'predictions/GMM_W2.csv');d['timestamp']=pd.to_datetime(d.timestamp)
    assert len(d)==4690
    threshold=float(d.threshold.iloc[0]);d['seed']=seed
    d['outcome']=np.select([(d.y_true==0)&d.y_pred,(d.y_true==1)&~d.y_pred,(d.y_true==1)&d.y_pred],['FP','FN','TP'],default='TN')
    d['reason_group']=d.outcome
    d.loc[(d.outcome=='FN')&(d.status=='warmup'),'reason_group']='FN_warmup'
    for clip,g in d[d.y_true==1].groupby('clip_id',sort=False):
        first=g.loc[g.y_pred,'sample_index'].min()
        scored_miss=g.index[(g.outcome=='FN')&(g.status=='scored')]
        d.loc[scored_miss,'reason_group']=np.where(g.loc[scored_miss,'sample_index']<first,'FN_before_first_alarm','FN_after_first_alarm') if np.isfinite(first) else 'FN_no_clip_alarm'
    d['margin']=d.anomaly_score-threshold
    # Recompute scores using the preserved estimator: validates alignment/scaling.
    zs=[];rows=[]
    for _,g in d.groupby('clip_id',sort=False):
        a=(g[S].to_numpy()-mu)/sd
        for i in range(2,len(g)):
            zs.append(a[i-2:i+1].ravel());rows.append(g.index[i])
    z=np.array(zs);rows=np.array(rows);nll=-model.score_samples(z)
    assert np.allclose(nll,d.loc[rows,'anomaly_score'],rtol=1e-9,atol=1e-9)
    assert np.array_equal(nll>threshold,d.loc[rows,'y_pred'])
    for sensor in S:
        d[f'{sensor}_step']=d.groupby('clip_id')[sensor].diff().abs()
        d[f'{sensor}_window_max_abs']=d.groupby('clip_id')[sensor].transform(lambda a:a.abs().rolling(3,min_periods=3).max())
    x,y,_=r.windows(data,'calibration',2,mu,sd);cal=np.c_[x,y]
    for j,sensor in enumerate(S):
        cols=[j,j+3,j+6];other=[k for k in range(9) if k not in cols]
        # Conditional score is a diagnostic, not an additive causal attribution.
        cond=nll-marginal_nll(model,z,other)
        cal_cond=-model.score_samples(cal)-marginal_nll(model,cal,other)
        d.loc[rows,f'{sensor}_conditional_nll']=cond
        d.loc[rows,f'{sensor}_conditional_percentile']=100*np.searchsorted(np.sort(cal_cond),cond,side='right')/len(cal_cond)
        marginal=marginal_nll(model,z,cols);cal_m=marginal_nll(model,cal,cols)
        d.loc[rows,f'{sensor}_marginal_percentile']=100*np.searchsorted(np.sort(cal_m),marginal,side='right')/len(cal_m)
    x,y,ti=r.windows(data,'train',2,mu,sd);tz=np.c_[x,y]
    nn=NearestNeighbors(n_neighbors=1).fit(tz);distance,index=nn.kneighbors(z)
    d.loc[rows,'nearest_training_distance']=distance[:,0]
    d.loc[rows,'nearest_train_row']=ti[index[:,0]]
    # Validate shared audit predictions, independently of saved full-score files.
    shared=Path(args.code_dir).parent/'results/sensor_audit'/f'all_sensors_seed{seed}_predictions.csv'
    if shared.exists():
        q=pd.read_csv(shared)
        assert list(zip(d.clip_id,d.sample_index))==list(zip(q.clip_id,q.sample_index))
        assert np.array_equal(d.y_pred,q.pred)
    fp=d.outcome.eq('FP');fn=d.outcome.eq('FN');a=d.y_true.eq(1)
    delays=d[a].groupby('clip_id').apply(lambda g:g.loc[g.y_pred,'elapsed_seconds'].min(),include_groups=False)
    m={'seed':seed,'FP':int(fp.sum()),'FN':int(fn.sum()),
       **{name:int(d.reason_group.eq(name).sum()) for name in ['FN_warmup','FN_before_first_alarm','FN_after_first_alarm','FN_no_clip_alarm']},
       'normal_alarm_clips':int(d[fp].clip_id.nunique()),'fault_no_alarm_clips':int(delays.isna().sum()),
       'early_0.5':float(delays.le(.5).mean()),'threshold':threshold}
    seed_metrics.append(m);tables.append(d);fits[seed]=(model,mu,sd)
    d.to_csv(OUT/f'seed{seed}_timestamp_diagnostics.csv',index=False)
    print(json.dumps(m),flush=True)
all_d=pd.concat(tables,ignore_index=True);d=tables[0]
pd.DataFrame(seed_metrics).to_csv(OUT/'seed_metrics.csv',index=False)
consensus=all_d.groupby(['clip_id','sample_index']).agg(y_true=('y_true','first'),alarm_votes=('y_pred','sum'),score_mean=('anomaly_score','mean'),score_sd=('anomaly_score','std')).reset_index()
consensus['all_seed_error']=((consensus.y_true==0)&(consensus.alarm_votes==3))|((consensus.y_true==1)&(consensus.alarm_votes==0))
consensus.to_csv(OUT/'seed_consensus.csv',index=False)
all_d[all_d.outcome.isin(['FP','FN'])].to_csv(OUT/'all_error_points.csv',index=False)

events=[]
for seed,df in enumerate(tables):
    for clip,g in df.groupby('clip_id',sort=False):
        error=g.reason_group.where(g.outcome.isin(['FP','FN']),'OK')
        groups=error.ne(error.shift()).cumsum()
        for _,h in g.groupby(groups):
            name=h.reason_group.iloc[0]
            if name in ['TP','TN']:continue
            events.append({'seed':seed,'clip_id':clip,'kind':name,'start_index':h.sample_index.iloc[0],
              'end_index':h.sample_index.iloc[-1],'points':len(h),'start_s':h.elapsed_seconds.iloc[0],
              'end_s':h.elapsed_seconds.iloc[-1],'max_margin':h.margin.max(),'min_margin':h.margin.min()})
pd.DataFrame(events).to_csv(OUT/'error_runs.csv',index=False)
clip_rows=[]
for clip,g in d.groupby('clip_id',sort=False):
    clip_rows.append({'clip_id':clip,'label':int(g.y_true.iloc[0]),'points':len(g),
      'FP':int(g.outcome.eq('FP').sum()),'FN_warmup':int(g.reason_group.eq('FN_warmup').sum()),
      'FN_scored':int((g.outcome.eq('FN')&g.status.eq('scored')).sum()),
      'first_alarm_s':g.loc[g.y_pred,'elapsed_seconds'].min(),
      'current_range':g.AI2_Current.max()-g.AI2_Current.min(),
      'vibration0_max_abs':g.AI0_Vibration.abs().max(),'vibration1_max_abs':g.AI1_Vibration.abs().max()})
clips=pd.DataFrame(clip_rows);clips.to_csv(OUT/'clip_summary_seed0.csv',index=False)
features=['margin','nearest_training_distance']+[s+k for s in S for k in ['_step','_window_max_abs','_conditional_percentile','_marginal_percentile']]
groups=d[d.status=='scored'].groupby('outcome')[features].agg(['median','min','max'])
groups.to_csv(OUT/'outcome_features_seed0.csv')

# Fixed bins derived from normal training sensor amplitudes, not error thresholds.
condition_rows=[]
for sensor in S:
    train=data.loc[data.split.eq('train'),sensor].abs();bounds=train.quantile([.5,.9,.99]).to_numpy()
    cats=pd.cut(d[sensor].abs(),[-np.inf,*bounds,np.inf],labels=['<=train p50','p50–p90','p90–p99','>train p99'])
    for label in ['normal','fault']:
        for cat in cats.cat.categories:
            mask=cats.eq(cat)&d.y_true.eq(0 if label=='normal' else 1)&d.status.eq('scored')
            g=d[mask]
            condition_rows.append({'sensor':sensor,'label':label,'bin':cat,'n':len(g),
                 'error_n':int((g.outcome=='FP' if label=='normal' else g.outcome=='FN').sum()),
                 'error_rate':float((g.outcome=='FP' if label=='normal' else g.outcome=='FN').mean()) if len(g) else np.nan})
pd.DataFrame(condition_rows).to_csv(OUT/'training_quantile_conditions_seed0.csv',index=False)

def plot_clip(clip):
    g=d[d.clip_id.eq(clip)];fig,axs=plt.subplots(4,1,figsize=(10,8),sharex=True)
    for ax,sensor in zip(axs[:3],S):
        ax.plot(g.elapsed_seconds,g[sensor],color='#315779',lw=1.3)
        for kind,color in [('FP','#cf7932'),('FN','#ba4141')]:
            m=g.outcome.eq(kind)&g.status.eq('scored')
            ax.scatter(g.loc[m,'elapsed_seconds'],g.loc[m,sensor],color=color,s=26,label=kind)
        ax.set_ylabel(sensor.replace('AI0_','Upper ').replace('AI1_','Lower ').replace('AI2_',''));ax.grid(alpha=.15)
    axs[3].plot(g.elapsed_seconds,g.anomaly_score,color='#315779',label='GMM NLL')
    axs[3].axhline(g.threshold.iloc[0],color='#333',ls='--',label='Fixed calibration threshold')
    for kind,color in [('FP','#cf7932'),('FN','#ba4141')]:
        m=g.outcome.eq(kind)&g.status.eq('scored');axs[3].scatter(g.loc[m,'elapsed_seconds'],g.loc[m,'anomaly_score'],color=color,s=26)
    for ax in axs:ax.axvspan(0,.2,color='#888',alpha=.12)
    axs[3].set_ylabel('Anomaly score');axs[3].set_xlabel('Seconds since clip start');axs[3].legend(frameon=False)
    axs[0].legend(frameon=False,ncol=2);fig.suptitle(f'{clip} | original 3-sensor GMM W=2, seed 0 | shaded: warmup')
    fig.tight_layout();fig.savefig(FIG/f'{clip}.png',dpi=170);plt.close(fig)

for clip in ['normal_0570','normal_0576','fault_0004','fault_0016','fault_0017','fault_0021']:plot_clip(clip)

# Error-to-correct nearest pairs: same label avoids mixing FP/TN and FN/TP.
pairs=[]
for kind,correct in [('FP','TN'),('FN','TP')]:
    pool=d[d.outcome.eq(correct)&d.status.eq('scored')]
    errs=d[d.outcome.eq(kind)&d.status.eq('scored')]
    # Rank by margin: largest FP and deepest scored FN are covered.
    errs=errs.sort_values('margin',ascending=(kind=='FN')).head(3)
    pool_rows=pool.index.to_numpy();lookup={row:i for i,row in enumerate(rows)}
    # `rows` here still has the same clip/sample ordering for every seed.
    model,mu,sd=fits[0]
    zz=[]
    for _,g in d.groupby('clip_id',sort=False):
        a=(g[S].to_numpy()-mu)/sd
        zz.extend([a[i-2:i+1].ravel() for i in range(2,len(g))])
    zz=np.array(zz)
    pair_nn=NearestNeighbors(n_neighbors=1).fit(zz[[lookup[i] for i in pool_rows]])
    for idx,e in errs.iterrows():
        dist,ix=pair_nn.kneighbors(zz[lookup[idx]][None,:]);partner=pool.loc[pool_rows[ix[0,0]]]
        pairs.append({'error_kind':kind,'clip':e.clip_id,'sample_index':int(e.sample_index),'score':e.anomaly_score,
          'margin':e.margin,'comparison_clip':partner.clip_id,'comparison_index':int(partner.sample_index),
          'comparison_score':partner.anomaly_score,'standardized_window_distance':float(dist[0,0])})
pd.DataFrame(pairs).to_csv(OUT/'nearest_correct_pairs_seed0.csv',index=False)

summary={'protocol':'original saved 3-sensor GMM W2, no fitting/threshold changes',
 'seed_metrics':seed_metrics,'seed0_outcome_medians':{k:v.to_dict() for k,v in d[d.status=='scored'].groupby('outcome')[features].median().iterrows()},
 'all_seed_FP_points':int(((consensus.y_true==0)&(consensus.alarm_votes==3)).sum()),
 'any_seed_FP_points':int(((consensus.y_true==0)&(consensus.alarm_votes>0)).sum()),
 'all_seed_scored_FN_points':int(((consensus.y_true==1)&(consensus.alarm_votes==0)&(consensus.sample_index>=2)).sum()),
 'provenance':[{k:v for k,v in p.items() if k!='path'} for p in provenance]}
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2))
print('MEDIANS',d[d.status=='scored'].groupby('outcome')[features].median().to_string(),flush=True)
print('CLIPS',clips[(clips.FP>0)|(clips.FN_scored>0)].to_string(index=False),flush=True)
