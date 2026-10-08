"""Aggregate fixed-split algorithm randomness, not fault-sample uncertainty."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from run_experiment import point_metrics, normal_clip_metrics

STOCHASTIC={'MLP32','LSTM32','GRU32','AE_Recon','GMM','IsolationForest'}
WIDTHS=[1,2,3,5,8,10,15,20,30]
SEEDS=[0,1,2]
METRICS=['F1','precision','recall','false_alarm_rate','miss_rate','normal_clip_alarm_rate',
         'TP','FP','TN','FN','normal_alert_clips','detected_clips','no_alert_clips',
         'median_delay_s','mean_delay_s','p90_delay_s','early_0.2s','early_0.5s','early_1s','early_2s',
         'common_F1','common_false_alarm_rate','common_miss_rate']


def cell(row,metric,pct=False,places=4):
    mean=row[f'{metric}_mean'];sd=row[f'{metric}_std']
    factor=100 if pct else 1
    if not np.isfinite(mean):return '—'
    if row['n']==1 and not row['stochastic']:
        return f'{mean*factor:.{places}f}'+('% (고정)' if pct else ' (고정)')
    if row['n']<3:return f'대기 ({row["n"]}/3)'
    return f'{mean*factor:.{places}f}'+('%' if pct else '')+f' ± {sd*factor:.{places}f}'+('%p' if pct else '')


def summarize(root,require_complete=False):
    root=Path(root);out=root/'seed_experiments';out.mkdir(exist_ok=True)
    records=[];manifest=[]
    for mode in ['natural','controlled']:
        baseline=root/'window_sweep'/mode/'results'
        baseline_config=json.loads((baseline/'experiment_config.json').read_text())
        for seed in SEEDS:
            folder=baseline if seed==0 else out/mode/f'seed_{seed}'/'results'
            path=folder/'timestamp_results.csv'
            if not path.exists():continue
            config=json.loads((folder/'experiment_config.json').read_text())
            if seed:
                assert config['training_seed']==seed
                assert {k:v for k,v in config.items() if k!='training_seed'}==baseline_config
            data=pd.read_csv(path);delays=pd.read_csv(folder/'delay_results.csv')
            for r in data.merge(delays,on=['model','window_size']).to_dict('records'):
                name=r['model'];w=int(r['window_size'])
                if seed and name not in STOCHASTIC:raise ValueError('Deterministic runs must not be replicated as seed observations')
                label=f'{name}_W{w}'
                params=json.loads((folder/'training'/f'{label}.json').read_text())
                if name in STOCHASTIC:assert params['seed']==seed
                pp=folder/'predictions'/f'{label}.csv';p=pd.read_csv(pp)
                assert len(p)==4690
                m=point_metrics(p)
                assert all(m[k]==r[k] for k in ['TP','FP','TN','FN'])
                assert normal_clip_metrics(p)['normal_alert_clips']==r['normal_alert_clips']
                ready=p.status.eq('scored');assert not p.loc[~ready,'y_pred'].any()
                np.testing.assert_array_equal(p.loc[ready,'y_pred'],p.loc[ready,'anomaly_score']>r['threshold'])
                common=point_metrics(p[p.sample_index>=30]);assert common['timestamps']==1281
                r.update({f'common_{k}':common[k] for k in ['F1','false_alarm_rate','miss_rate']})
                for t in ['0.2','0.5','1','2']:r[f'early_{t}s']=r[f'detected_by_{t}s']/21
                r.update({'mode':mode,'seed':seed if name in STOCHASTIC else None,'stochastic':name in STOCHASTIC})
                records.append(r)
                manifest.append({'mode':mode,'model':name,'window_size':w,'seed':r['seed'],
                                 'source':str(pp.relative_to(root)),'sha256':hashlib.sha256(pp.read_bytes()).hexdigest()})
    raw=pd.DataFrame(records)
    assert not raw.duplicated(['mode','model','window_size','seed']).any()
    aggregated=[]
    for (mode,name,w),g in raw.groupby(['mode','model','window_size'],sort=False):
        stochastic=name in STOCHASTIC
        if require_complete:assert len(g)==(3 if stochastic else 1)
        row={'mode':mode,'model':name,'window_size':int(w),'stochastic':stochastic,'n':len(g)}
        for metric in METRICS:
            v=g[metric].dropna().to_numpy(dtype=float)
            row[f'{metric}_mean']=float(v.mean()) if len(v) else np.nan
            row[f'{metric}_std']=float(v.std(ddof=1)) if len(v)>1 else np.nan
        aggregated.append(row)
    summary=pd.DataFrame(aggregated)
    complete=len(raw)==522 and all(r['n']==(3 if r['stochastic'] else 1) for r in aggregated)
    if require_complete:assert complete
    raw.to_csv(out/'seed_runs.csv',index=False);summary.to_csv(out/'seed_summary.csv',index=False)
    (out/'seed_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    lines=['# Seed 0·1·2 반복 실험: 평균 ± 표준편차','',
           f'상태: {"완료" if complete else "실행 중"}. 독립 실행 기록 {len(raw)}/522개. 기존 seed=0 기록을 유지하고 seed=1·2의 216조합을 추가 학습했다.' if complete else f'상태: 실행 중. 현재 실행 기록 {len(raw)}/522개.', '',
           '## 반복이 필요한 방법과 집계 규칙','',
           '| 방법 | 반복 | 변동 원인 |', '|---|---|---|',
           '| MLP32·LSTM32·GRU32·AE_Recon | seed 0·1·2 각각 학습 | 가중치 초기화·배치 순서 |',
           '| GMM | seed 0·1·2, 각 실행 n_init=2 유지 | 혼합 분포 초기화·국소 최적해 |',
           '| IsolationForest | seed 0·1·2 | 학습 샘플링·무작위 분할 |',
           '| Persistence·AR·VAR_Ridge·AR_Relation | 1회 | 본 구현은 고정된 데이터의 결정론적 예측·닫힌해 ridge |',
           '| PCA_Recon·Mahalanobis·kNN20·KDE·LOF20·OCSVM·RobustMAD | 1회 | full SVD·공분산 추정·이웃/밀도/경계 계산은 본 설정에서 결정론적 |','',
           '- 평균과 표준편차는 각 seed의 지표를 먼저 구한 뒤 집계한다. 평균 혼동 행렬에서 F1을 다시 계산하지 않는다.',
           '- 표준편차는 표본 표준편차 s=√[Σ(metric−평균)²/(3−1)], ddof=1이다. 단위가 비율이면 평균은 %, SD는 퍼센트포인트(%p)로 표기한다.',
           '- 결정론적 방법은 반복하지 않고 `고정`으로 표기한다. 단일 실행을 세 번 복제해 SD=0으로 제시하지 않는다.',
           '- 데이터 분할·정규화·W·학습 규칙은 고정한다. 각 seed의 임계값은 그 seed 모델의 정상 보정 점수 99백분위수에서 다시 계산한다.',
           '- seed=0은 이미 저장된 동일 프로토콜의 W 확장 결과를 재사용했다. seed=1·2의 원본 판정·학습 기록·모델은 별도 폴더에 보존한다.',
           '- 각 seed를 독립 고장 사례로 취급하지 않는다. SD는 동일 데이터에서 알고리즘 초기화/샘플링에 따른 변동이며 일반화 신뢰구간이 아니다.',
           '- 테스트 F1로 가장 좋은 seed를 골라 보고하지 않는다. 세 seed 모두 포함한다. W별 평균 최고값도 탐색 관찰이며 최종 W 선정과 구분한다.','',
           '## W=2 자연 운용: 모든 테스트 시점과 조기경보','',
           '| 방법 | 실행 수 | F1 | 시점 오탐률 | 시점 미탐률 | 0.2초 이내 탐지율 | 0.5초 이내 탐지율 | 정상 클립 경보율 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary[(summary['mode']=='natural')&(summary.window_size==2)].to_dict('records'):
        vals=[cell(r,'F1')]+[cell(r,k,pct=True,places=2) for k in ['false_alarm_rate','miss_rate','early_0.2s','early_0.5s','normal_clip_alarm_rate']]
        lines.append(f'| {r["model"]} | {r["n"]} | '+' | '.join(vals)+' |')
    for mode,metric,title in [('natural','F1','전체 시점 운용 F1'),('natural','common_F1','평가 시점만 통일한 F1'),('controlled','common_F1','학습부터 통제한 공통 시점 F1')]:
        lines+=['',f'## {title}: W별 평균 ± SD','',
                '| 방법 | '+' | '.join(f'W={w}' for w in WIDTHS)+' |', '|---|'+'---:|'*len(WIDTHS)]
        subset=summary[summary['mode']==mode]
        for name,g in subset.groupby('model',sort=False):
            lookup={r['window_size']:r for r in g.to_dict('records')}
            lines.append('| '+name+' | '+' | '.join(cell(lookup[w],metric) if w in lookup else '대기' for w in WIDTHS)+' |')
    lines+=['','## 자연 운용 지연과 끝까지 미탐: 평균 ± SD','',
            '지연 중앙값·P90은 각 seed에서 경보한 클립만으로 구한 값의 평균±SD다. 경보하지 않은 클립 수와 함께 읽는다. 클립 시작 기준 지연이며 실제 고장 onset부터의 지연은 아니다.','',
            '| 방법 | W | 최초 판단(초) | 끝까지 미탐 클립 수 | 지연 중앙값(초) | 지연 P90(초) |',
            '|---|---:|---:|---:|---:|---:|']
    for r in summary[(summary['mode']=='natural')&summary.model.isin(['GMM','Mahalanobis','LSTM32','GRU32'])].to_dict('records'):
        lines.append(f'| {r["model"]} | {r["window_size"]} | {r["window_size"]*.1:.1f} | {cell(r,"no_alert_clips",places=2)} | {cell(r,"median_delay_s")} | {cell(r,"p90_delay_s")} |')
    lines+=['','## 데이터와 해석 범위','',
            '- 자연 운용은 정상 4,090+이상 600=4,690시점 전체를 평가하고 준비 구간은 경보 없음으로 남긴다.',
            '- 공통 시점은 index≥30인 정상 1,135+이상 146=1,281시점이다. 이상 10클립만 남으므로 이 결과의 높은 F1은 전체 조기경보 성능이 아니다.',
            '- 각 seed에서도 정상 클립 경보 예산을 동일하게 맞춘 것은 아니다. 임계값 산출 규칙이 같다는 것과 작업 중단 빈도가 같다는 것은 다르다.',
            '- 기록이 정상/이상 날짜 각각 하나이고 이상 21클립이 한 기록에서 나온 제한은 반복 seed로 해소되지 않는다.',
            '- 상세 CSV: `seed_experiments/seed_runs.csv`·`seed_summary.csv`. 원본 경로·해시: `seed_sources.json`. 원본 seed=0은 `window_sweep/{natural,controlled}/results`, seed=1·2는 `seed_experiments/{natural,controlled}/seed_{1,2}/results`.','']
    path=root/'SEED_RESULTS.md';temp=path.with_suffix('.md.tmp');temp.write_text('\n'.join(lines),encoding='utf-8');temp.replace(path)
    return raw,summary,complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent);p.add_argument('--require-complete',action='store_true')
    args=p.parse_args();summarize(args.root,args.require_complete)
