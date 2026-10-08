"""All-timestamp confusion metrics and clip-start-to-first-alarm delay."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def evaluate(df):
    y=df.y_true.to_numpy(dtype=int)
    if not set(y).issubset({0,1}): raise ValueError('Binary labels required')
    if df.pred.dtype!=bool: raise ValueError('pred must be True/False')
    pred=df.pred.to_numpy();y=y.astype(bool)
    tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    alarms=[]
    for cid,g in df.groupby('clip_id',sort=False):
        if g.y_true.nunique()!=1:raise ValueError('Expected one file-level label per clip')
        hits=g.loc[g.pred,'elapsed_seconds']
        alarms.append({'clip_id':cid,'y_true':int(g.y_true.iloc[0]),'detected':bool(len(hits)),
                       'first_alarm_seconds':float(hits.iloc[0]) if len(hits) else np.nan})
    delays=pd.DataFrame(alarms);fault=delays[delays.y_true.eq(1)];normal=delays[delays.y_true.eq(0)]
    valid=fault.first_alarm_seconds.dropna()
    divide=lambda a,b:float(a/b) if b else np.nan
    result={'seed':int(df.seed.iloc[0]),'mode':df['mode'].iloc[0],'timestamps':len(df),
            'TP':tp,'FP':fp,'TN':tn,'FN':fn,'F1':divide(2*tp,2*tp+fp+fn),
            'precision':divide(tp,tp+fp),'recall':divide(tp,tp+fn),'false_alarm_rate':divide(fp,fp+tn),'miss_rate':divide(fn,tp+fn),
            'normal_alarm_clips':int(normal.detected.sum()),'fault_clips':len(fault),'fault_detected_clips':int(fault.detected.sum()),
            'delay_mean_s':float(valid.mean()),'delay_p90_s':float(valid.quantile(.9)),
            'detected_by_0.2s':float(fault.first_alarm_seconds.le(.2+1e-9).mean()),
            'detected_by_0.5s':float(fault.first_alarm_seconds.le(.5+1e-9).mean())}
    return result,delays

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,nargs='+',default=[ROOT/'predictions/test_predictions.csv'])
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/evaluation')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
    for path in args.predictions:
        df=pd.read_csv(path);metric,delays=evaluate(df);rows.append(metric)
        delays.to_csv(args.output_dir/f"first_alarm_{metric['mode']}_seed{metric['seed']}.csv",index=False)
    table=pd.DataFrame(rows)
    if table[['mode','seed']].duplicated().any():raise ValueError('One prediction file per mode/seed required')
    table.to_csv(args.output_dir/'metrics_by_seed.csv',index=False)
    summary=table.groupby('mode').agg({c:['mean','std'] for c in table.select_dtypes(include='number').columns if c!='seed'})
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(args.output_dir/'summary.csv')
    print(table[['mode','seed','F1','FP','FN','delay_mean_s']].to_string(index=False))

if __name__=='__main__': main()
