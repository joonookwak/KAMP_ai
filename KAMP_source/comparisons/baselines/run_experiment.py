"""Normal-only causal anomaly detection; common all-timestamp evaluation."""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from models import MODELS

SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
WINDOWS = [2, 3, 5, 10]
DEFAULT_DATA = str(Path(__file__).resolve().parents[1] / "data")


def prepare(data_dir):
    frames, provenance = [], []
    for filename, source, state in [("press_data_normal.csv", "normal", 0), ("outlier_data.csv", "fault", 1)]:
        path = data_dir / filename
        if not path.exists() and (data_dir / 'raw' / filename).exists():
            path = data_dir / 'raw' / filename
        d = pd.read_csv(path)
        d["timestamp"] = pd.to_datetime(d.TimeStamp)
        if not d.Equipment_state.eq(state).all() or not d.timestamp.is_monotonic_increasing:
            raise ValueError("Unexpected state or timestamp order")
        if d[SENSORS].isna().any().any():
            raise ValueError("Missing sensor values")
        duplicate_times = d[d.timestamp.duplicated(keep=False)]
        for _, g in duplicate_times.groupby("timestamp"):
            if len(g[SENSORS+['Equipment_state']].drop_duplicates()) != 1:
                raise ValueError("Conflicting samples at the same timestamp")
        raw_n = len(d)
        d = d.drop_duplicates("timestamp").reset_index(drop=True)
        numbers = d.timestamp.diff().dt.total_seconds().gt(.5).cumsum()+1
        d["clip_id"] = numbers.map(lambda i: f"{source}_{i:04d}")
        d["sample_index"] = d.groupby("clip_id", sort=False).cumcount()
        d["elapsed_seconds"] = (d.timestamp-d.groupby("clip_id").timestamp.transform("first")).dt.total_seconds()
        d["source"] = source
        d["y_true"] = d.Equipment_state.astype(int)
        d["source_row"] = d.iloc[:, 0]
        frames.append(d[["source", "source_row", "clip_id", "sample_index", "timestamp",
                         "elapsed_seconds", *SENSORS, "y_true"]])
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "raw_rows": raw_n, "clean_rows": len(d), "clips": d.clip_id.nunique()})
    samples = pd.concat(frames, ignore_index=True)
    segments = samples.groupby("clip_id", sort=False).agg(
        source=("source", "first"), n_samples=("timestamp", "size"),
        start_time=("timestamp", "first"), end_time=("timestamp", "last")).reset_index()
    normal_ids = segments.loc[segments.source=='normal', 'clip_id'].tolist()
    if len(normal_ids)!=599 or (segments.source=='fault').sum()!=21:
        raise ValueError("Dataset does not match the agreed 599 normal / 21 fault clips")
    split_ids = {"train": normal_ids[:305], "validation": normal_ids[305:359],
                 "calibration": normal_ids[359:479], "normal_test": normal_ids[479:],
                 "fault_test": segments.loc[segments.source=='fault','clip_id'].tolist()}
    mapping = {sid:split for split, ids in split_ids.items() for sid in ids}
    samples["split"] = samples.clip_id.map(mapping)
    segments["split"] = segments.clip_id.map(mapping)
    return samples, segments, provenance


def windows(samples, split, width, mean, scale, minimum_index=0):
    xs, ys, indices = [], [], []
    for _, g in samples[samples.split==split].groupby("clip_id", sort=False):
        z = (g[SENSORS].to_numpy()-mean)/scale
        for target in range(max(width,minimum_index), len(g)):
            xs.append(z[target-width:target].reshape(-1))
            ys.append(z[target]); indices.append(g.index[target])
    return (np.asarray(xs, dtype=float).reshape(-1, width*3),
            np.asarray(ys, dtype=float).reshape(-1, 3), np.asarray(indices, dtype=int))


def point_metrics(frame):
    actual = frame.y_true.to_numpy().astype(bool)
    predicted = frame.y_pred.to_numpy().astype(bool)
    tp, fp = int(np.sum(actual & predicted)), int(np.sum(~actual & predicted))
    tn, fn = int(np.sum(~actual & ~predicted)), int(np.sum(actual & ~predicted))
    div = lambda a,b: float(a/b) if b else None
    return {"timestamps": len(frame), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": div(tp,tp+fp), "recall": div(tp,tp+fn),
            "F1": div(2*tp,2*tp+fp+fn), "false_alarm_rate": div(fp,fp+tn),
            "miss_rate": div(fn,tp+fn), "warmup_normal": int(np.sum(~actual & (frame.status=='warmup'))),
            "warmup_fault": int(np.sum(actual & (frame.status=='warmup')))}


def delays(predictions):
    rows=[]
    for sid,g in predictions[predictions.source=='fault'].groupby('clip_id',sort=False):
        alarm=g[g.y_pred]
        rows.append({"clip_id":sid, "start_time":g.timestamp.iloc[0],
                     "first_alert_time":alarm.timestamp.iloc[0] if len(alarm) else None,
                     "delay_seconds":float(alarm.elapsed_seconds.iloc[0]) if len(alarm) else None,
                     "detected":bool(len(alarm)), "scorable":bool((g.status=='scored').any())})
    table=pd.DataFrame(rows);valid=table.loc[table.detected,'delay_seconds'].to_numpy(dtype=float)
    metrics={"fault_clips":len(table),"detected_clips":int(table.detected.sum()),
             "no_alert_clips":int((~table.detected).sum()),
             "unscorable_clips":int((~table.scorable).sum()),
             "median_delay_s":float(np.median(valid)) if len(valid) else None,
             "mean_delay_s":float(np.mean(valid)) if len(valid) else None,
             "p90_delay_s":float(np.quantile(valid,.9)) if len(valid) else None}
    for t in [.2,.5,1.,2.]:
        metrics[f"detected_by_{t:g}s"]=int(((table.delay_seconds<=t+1e-9)&table.detected).sum())
    return table,metrics


def normal_clip_metrics(predictions):
    alarms=predictions[predictions.source=='normal'].groupby('clip_id').y_pred.any()
    return {'normal_clips':len(alarms),'normal_alert_clips':int(alarms.sum()),
            'normal_clip_alarm_rate':float(alarms.mean())}


def score_model(model,x,y):
    scores=(model.score(x,y) if hasattr(model,'score') else
            np.mean((model.predict(x)-y)**2,axis=1))
    scores=np.asarray(scores,dtype=float)
    if scores.shape!=(len(y),) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite scalars, higher = more anomalous')
    return scores


def write_report(out_dir, results, delay_results, planned, complete=False):
    """Update the human-readable report after every completed method/window."""
    table=pd.DataFrame(results)
    delays_table=pd.DataFrame(delay_results)
    pct=lambda v:f'{float(v)*100:.2f}%'
    number=lambda v:'—' if v is None or pd.isna(v) else f'{float(v):.3f}'
    done={(r['model'],int(r['window_size'])):r for r in results}
    delay_map={(r['model'],int(r['window_size'])):r for r in delay_results}
    keys=sorted(set(done)|set(planned),key=lambda k:(k[0],k[1]))
    now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    config_path=out_dir/'experiment_config.json'
    minimum_index=json.loads(config_path.read_text()).get('minimum_target_index',0) if config_path.exists() else 0
    lines=['# 타임스탬프 단위 조기 이상탐지 실험', '',
           f'마지막 갱신: {now}', '',
           f'상태: {"현재 요청한 실험 완료" if complete else "실행 중 — 조합별 완료 즉시 갱신"}', '',
           '## 공통 실험 규칙', '',
           f'- 공통 시작 인덱스={minimum_index}. 0이면 방법별 W 이후부터 학습·보정·판단. 양수이면 모든 분할에서 index≥max(W,공통 인덱스)인 동일 시점만 사용하며 그 이전은 경보 없음이다.',
           '- 정상 클립: 학습 305 / 검증 54 / 임계값 보정 120 / 테스트 120. 이상 21클립은 테스트에만 사용.',
           '- 판단 시점 t에서 과거 W개와 현재 3센서까지 사용 가능. 예측기는 과거 W개로 현재값을 예측해 오차를 계산한다. 재구축·분포 방법은 과거 W개+현재값을 직접 점수화한다. 미래값은 사용하지 않는다. stride=1.',
           '- 정규화는 정상 학습값으로만 계산. 예측·재구축·거리·밀도 점수는 방법별로 정의하며 큰 점수일수록 이상이다.',
           '- 임계값: 정상 보정 데이터의 예측 가능한 타임스탬프 점수 99백분위수. 점수 > 임계값이면 True.',
           '- 모든 테스트 타임스탬프가 전체 평가에 포함. 준비 구간 max(W,공통 인덱스)개는 경보 없음(False)이며 이상 라벨이면 FN. 통제 실험은 준비 구간을 제외한 동일 평가 시점의 성능도 별도 보고한다.',
           '- 경보 유지(latch), 정답에 따른 소급 보정(point adjustment), 클립 경계 보간을 사용하지 않음.',
           '- 모델 설정·학습 종료는 정상 데이터로만 결정. 기존 예측기의 W 선택은 정상 검증 MSE 기반. 새 탐지기는 모든 W를 보고하며 테스트 지표로 W를 선택하지 않음.', '',
           '- 테스트 전체는 정상 4,090시점 + 이상 600시점 = 4,690시점. 모든 방법에서 같은 정답과 분모를 사용한다.', '',
           '## 비교 방법', '',
           '- Persistence: 직전값 예측. VAR_Ridge: 여러 센서의 과거값을 함께 쓰는 선형 예측. MLP32: 은닉층 32개 신경망.',
           '- AR: 각 센서의 과거값으로 해당 센서만 예측. AR_Relation: AR 예측 + 전류·진동의 비선형 관계를 이용한 잔차 회귀 보정.',
           '- AR_Relation의 다음 전류 입력은 AR 예측값이다. 실제 다음 전류값은 입력하지 않는다. 관계 회귀는 인과관계 증명이 아니다.',
           '- LSTM32·GRU32: 실제 단방향 순환층 1개, 은닉 크기 32, 선형 3센서 출력. 창마다 상태 초기화, 정상 검증으로 조기 종료.', '',
           '- PCA_Recon·AE_Recon: 과거+현재 관측 창 전체의 재구축 MSE. Mahalanobis·kNN20: 정상 분포/이웃에서의 거리.',
           '- GMM·KDE: 음의 로그 밀도. LOF20: 정상 학습 이웃 대비 국소 밀도 이상도. IsolationForest: 격리 점수. OCSVM: 정상 경계 밖의 정도.',
           '- RobustMAD: 현재 세 센서만의 robust z 최대값. 다른 방법과 같은 W 준비 구간을 적용한 정적 기준선.', '',
           '## 주 결과: 모든 테스트 타임스탬프', '',
           '| 방법 | W | TP | FP | TN | FN | F1 | 오탐률 FP/(FP+TN) | 미탐률 FN/(TP+FN) | 정상/이상 준비 시점 수 | 상태 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for key in keys:
        if key not in done:
            lines.append(f'| {key[0]} | {key[1]} | — | — | — | — | — | — | — | — | 대기 |')
            continue
        r=done[key]
        lines.append(f'| {key[0]} | {key[1]} | {r["TP"]} | {r["FP"]} | {r["TN"]} | {r["FN"]} | '
                     f'{number(r["F1"])} | {pct(r["false_alarm_rate"])} | {pct(r["miss_rate"])} | '
                     f'{r["warmup_normal"]}/{r["warmup_fault"]} | 완료 |')
    lines+=['', '## 조기경보와 정상 작업 중단 비교', '',
            '이상 시간별 탐지율의 분모는 미탐·판단 불가를 포함한 21클립이다. 정상 클립 경보율은 정상 120클립 중 한 번이라도 True인 비율이다. 모든 방법은 정상 보정 시점 점수의 99백분위수를 쓰지만 정상 클립 경보율이 같도록 맞춘 것은 아니다.', '',
            '| 방법 | W | 0.2초 이내 탐지율 | 0.5초 이내 탐지율 | 1초 이내 탐지율 | 정상 경보 클립/120 | 정상 클립 경보율 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for key in keys:
        if key not in done or key not in delay_map:continue
        r=done[key];d=delay_map[key]
        counts=[pct(d[f'detected_by_{t}s']/d['fault_clips']) for t in ['0.2','0.5','1']]
        nc=r.get('normal_alert_clips');nr=r.get('normal_clip_alarm_rate')
        lines.append(f'| {key[0]} | {key[1]} | '+ ' | '.join(counts)+f' | {int(nc) if nc is not None and not pd.isna(nc) else "—"}/120 | {pct(nr) if nr is not None and not pd.isna(nr) else "—"} |')
    lines+=['', '## 첫 경보 지연: 타임스탬프 평가의 보조 결과', '',
            '지연은 클립 시작부터 첫 True 시점까지다. 지연 통계는 탐지한 클립만의 값이므로 미탐 수와 함께 해석한다. 판단 불가 클립은 미탐 클립 수에 포함한다.', '',
            '| 방법 | W | 첫 판단 가능(초) | 첫 경보 있음/21 | 끝까지 경보 없음 | 판단 불가 | 지연 중앙값(초) | 지연 평균(초) | 지연 P90(초) | 0.2초 이내 | 0.5초 이내 | 1초 이내 | 2초 이내 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in keys:
        if key not in delay_map:continue
        r=delay_map[key]
        lines.append(f'| {key[0]} | {key[1]} | {max(key[1],minimum_index)*.1:.1f} | {r["detected_clips"]}/21 | '
                     f'{r["no_alert_clips"]} | {r["unscorable_clips"]} | {number(r["median_delay_s"])} | '
                     f'{number(r["mean_delay_s"])} | {number(r["p90_delay_s"])} | '
                     f'{r["detected_by_0.2s"]} | {r["detected_by_0.5s"]} | {r["detected_by_1s"]} | {r["detected_by_2s"]} |')
    lines+=['','## 모델 검증 기록','',f'공통 검증 시점(sample_index ≥ {max(WINDOWS)})의 예측 MSE로 기존 예측기의 윈도우를 선택한다. 재구축/밀도 점수와 예측 MSE는 다른 척도이므로 새 방법의 W는 선택하지 않는다. 재구축/밀도/거리 점수와 설정은 학습 JSON에 기록한다.', '',
            '| 방법 | W | 정상 검증 MSE(공통 시점) | 경보 임계값 | 정상 검증으로 선택한 W |', '|---|---:|---:|---:|---|']
    for key in keys:
        if key in done:
            r=done[key];chosen=r.get('selected_by_normal_validation',False)
            mark='선택' if chosen is True or chosen==np.bool_(True) else '—'
            lines.append(f'| {key[0]} | {key[1]} | {number(r["validation_common_mse"])} | {number(r["threshold"])} | {mark} |')
    lines+=['', '## 해석 범위와 파일', '',
            '- 이상 파일의 모든 타임스탬프 라벨이 1인 조건에서 계산한 F1·오탐·미탐이다. 실제 이상 신호의 세밀한 시작 시각은 별도 검증되지 않았다.',
            '- 이상 21클립은 하나의 이상 기록에서 나왔다. 이미 개발 과정에서 살펴본 데이터이므로 독립 고장 사례의 일반화 성능을 증명하는 최종 평가가 아니다.',
            '- 윈도우가 길면 준비 구간의 이상 시점도 FN에 포함되므로 최초 판단까지의 대기 비용이 주 결과에 반영된다.',
            '- Persistence는 마지막 관측값을 예측한다. 윈도우 비교를 위해 다른 모델과 동일하게 W개 준비 구간을 적용한다.',
            '- 결과 CSV/JSON과 상세 예측은 `results/` 아래에 저장된다. `metrics.json`에 준비 구간을 제외한 보조 지표도 포함된다.',
            '- 방법을 추가할 때 기존 이름을 재사용하지 않고 새 이름으로 등록한다. 같은 프로토콜의 기존 결과는 유지하며 해당 방법·윈도우 결과만 갱신한다.', '']
    if minimum_index==0 and all((name,w) in done for name in ['AR','AR_Relation','LSTM32','GRU32'] for w in [2,3,5,10]):
        lines+=['## 추가 실험 해석', '',
                f'- W=2에서 독립 AR F1={done[("AR",2)]["F1"]:.3f}, 관계 회귀 보정 AR F1={done[("AR_Relation",2)]["F1"]:.3f}. 짧은 입력에서는 변수 관계 보정이 도움이 됐다.',
                f'- W=2에서 LSTM F1={done[("LSTM32",2)]["F1"]:.3f}, GRU F1={done[("GRU32",2)]["F1"]:.3f}. 첫 판단 가능 시각은 모두 0.2초다.',
                '- W=10에서는 관계 회귀 보정이 독립 AR보다 낮았다. 관계 항을 추가한다고 항상 성능이 오르지는 않았다.',
                '- 이번 단일 seed 결과에서 1초 윈도우 LSTM·GRU는 기존 VAR·MLP를 넘지 못했다. 모델 복잡도 증가가 일관된 개선으로 이어지지 않았다.',
                '- 조기탐지 후보 비교에는 W=2의 LSTM·GRU와 VAR·관계 보정 AR을 함께 둔다. 정상 검증 MSE가 선택한 W=10과 조기 경보 요구는 서로 다른 평가 목표다.',
                '- 0.2초 이내 탐지 수는 최초 판단 시점에 바로 경보한 클립 수다. 이는 모든 이상 타임스탬프를 맞혔다는 뜻이 아니다.',
                '- 신경망은 seed=0 한 번의 결과다. 근소한 F1 차이로 우열을 확정하지 않는다. 다음 검증에서는 여러 seed와 센서별 제거 실험으로 변동성과 전류 의존성을 확인할 수 있다.', '']
    notes=out_dir.parent/'SURVEY_FINDINGS.md'
    if notes.exists():lines+=['## 다양한 이상탐지 방법 비교 해석', '',notes.read_text(encoding='utf-8'),'']
    window_notes=out_dir.parent/'WINDOW_ANALYSIS.md'
    if window_notes.exists():lines+=['## W 범위 확장과 비교 통제', '',window_notes.read_text(encoding='utf-8'),'']
    seed_notes=out_dir.parent/'SEED_RESULTS.md'
    if seed_notes.exists():lines+=['## Seed 반복 실험 최신 결과', '',seed_notes.read_text(encoding='utf-8'),'']
    path=out_dir.parent/'EXPERIMENT_RESULTS.md'
    temporary=path.with_suffix('.md.tmp');temporary.write_text('\n'.join(lines),encoding='utf-8');temporary.replace(path)


def run(data_dir,out_dir,model_names=None,minimum_index=0,seed=0):
    out_dir.mkdir(parents=True,exist_ok=True)
    for folder in ['predictions','models','training']:(out_dir/folder).mkdir(exist_ok=True)
    samples, segments, provenance=prepare(data_dir)
    samples.to_csv(out_dir/'samples.csv',index=False)
    segments.to_csv(out_dir/'clip_splits.csv',index=False)
    training_values=samples.loc[samples.split=='train',SENSORS].to_numpy()
    mean,scale=training_values.mean(0),training_values.std(0)
    if np.any(scale<=0):raise ValueError('Zero training scale')
    config={"protocol_version":"timestamp_ad_v2", "sensor_order":SENSORS,
            "source_files":provenance, "gap_seconds":.5,"window_sizes":WINDOWS,
            "train_stride":1,"test_stride":1,"forecast_horizon_samples":1,
            "calibration_quantile":.99,"threshold_comparison":"strict >",
            "score":"method-specific causal score; definitions in training JSON",
            "warmup":"no alarm (False); included in all-timestamp metrics; anomaly warmup counts as FN",
            "alarm_policy":"per-timestamp threshold; no latch or retrospective point adjustment",
            "split_clips":segments.groupby('split').size().to_dict(),
            "normalization_mean":mean.tolist(),"normalization_scale":scale.tolist(),
            "model_selection":"normal-only hyperparameters; forecasting W uses normal validation MSE; detector W not selected",
            "interpretation":"fault file is one episode; already examined during development; exploratory comparison"}
    if WINDOWS!=[2,3,5,10] or minimum_index:
        config['protocol_version']='timestamp_ad_window_v3'
        config['minimum_target_index']=minimum_index
    if seed:
        config['training_seed']=seed
    config_path=out_dir/'experiment_config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text())
        if old!=config:
            excluded={'protocol_version','score','model_selection','normalization_mean','normalization_scale'}
            compatible=(old.get('protocol_version')=='timestamp_forecast_v1' and
                        {k:v for k,v in old.items() if k not in excluded}=={k:v for k,v in config.items() if k not in excluded} and
                        all(np.allclose(old[k],config[k],rtol=1e-12,atol=1e-12) for k in ['normalization_mean','normalization_scale']))
            if not compatible:raise ValueError('Existing results use a different protocol; choose a new output directory')
            (out_dir/'experiment_config_forecast_v1.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    load_records=lambda name:pd.read_csv(out_dir/name).to_dict('records') if (out_dir/name).exists() else []
    result_rows=load_records('timestamp_results.csv');delay_rows=load_records('delay_results.csv')
    all_details=json.loads((out_dir/'metrics.json').read_text()) if (out_dir/'metrics.json').exists() else {}
    old_delay=pd.read_csv(out_dir/'detection_delay.csv') if (out_dir/'detection_delay.csv').exists() else pd.DataFrame()
    # Add new metrics to prior results by reading the original per-timestamp decisions.
    for row in result_rows:
        name=row['model'];width=int(row['window_size'])
        row.update(normal_clip_metrics(pd.read_csv(out_dir/'predictions'/f'{name}_W{width}.csv')))
        row['family']=getattr(MODELS[name],'family','forecast')
        row['score_type']=getattr(MODELS[name],'score_description','next-sample standardized MSE')
    delay_details=[old_delay] if len(old_delay) else []
    model_names=list(MODELS) if model_names is None else model_names
    planned=[(name,width) for name in model_names for width in WINDOWS]
    write_report(out_dir,result_rows,delay_rows,planned)
    for width in WINDOWS:
        datasets={s:windows(samples,s,width,mean,scale,minimum_index) for s in
                  ['train','validation','calibration','normal_test','fault_test']}
        x,y,_=datasets['train'];vx,vy,vi=datasets['validation']
        common=samples.loc[vi,'sample_index'].to_numpy()>=max(WINDOWS)
        for name in model_names:
            factory=MODELS[name]
            label=f'{name}_W{width}'
            model=factory()
            if hasattr(model,'seed'):model.seed=seed
            started=time.perf_counter();params=model.fit(x,y,vx,vy)
            fit_seconds=time.perf_counter()-started
            family=getattr(model,'family','forecast')
            validation_scores=score_model(model,vx,vy)
            validation_mse=float(validation_scores.mean()) if family=='forecast' else np.nan
            common_mse=float(validation_scores[common].mean()) if family=='forecast' else np.nan
            params.update({'family':family,'score_type':getattr(model,'score_description','next-sample standardized MSE'),
                           'fit_seconds':fit_seconds,'validation_mean_score':float(validation_scores.mean()),
                           'observed_window_samples':width+1,'future_samples_used':0,
                           'minimum_target_index':minimum_index,'train_windows':len(x),
                           'validation_windows':len(vx),'calibration_windows':len(datasets['calibration'][0])})
            cx,cy,_=datasets['calibration']
            calibration_scores=score_model(model,cx,cy)
            threshold=float(np.quantile(calibration_scores,.99))
            predictions=samples[samples.split.isin(['normal_test','fault_test'])].copy()
            predictions['model']=name;predictions['window_size']=width
            predictions['family']=family
            predictions['status']='warmup';predictions['y_pred']=False
            predictions['anomaly_score']=np.nan;predictions['threshold']=threshold
            for sensor in SENSORS:
                predictions['pred_'+sensor]=np.nan
                predictions['error_standardized_'+sensor]=np.nan
                predictions['recon_'+sensor]=np.nan
            for split in ['normal_test','fault_test']:
                tx,ty,ti=datasets[split]
                scores=score_model(model,tx,ty)
                predictions.loc[ti,'status']='scored'
                predictions.loc[ti,'anomaly_score']=scores
                predictions.loc[ti,'y_pred']=scores>threshold
                if family=='forecast':
                    yp=model.predict(tx);errors=ty-yp
                    for j,sensor in enumerate(SENSORS):
                        predictions.loc[ti,'pred_'+sensor]=yp[:,j]*scale[j]+mean[j]
                        predictions.loc[ti,'error_standardized_'+sensor]=errors[:,j]
                elif hasattr(model,'reconstruct'):
                    current_recon=model.reconstruct(tx,ty)[:,-3:]
                    for j,sensor in enumerate(SENSORS):
                        predictions.loc[ti,'recon_'+sensor]=current_recon[:,j]*scale[j]+mean[j]
            assert len(predictions)==len(samples[samples.split.isin(['normal_test','fault_test'])])
            assert not predictions.loc[predictions.status=='warmup','y_pred'].any()
            predictions.to_csv(out_dir/'predictions'/f'{label}.csv',index=False)
            metrics=point_metrics(predictions);scored_metrics=point_metrics(predictions[predictions.status=='scored'])
            metrics.update(normal_clip_metrics(predictions))
            detail,delay=delays(predictions);detail.insert(0,'window_size',width);detail.insert(0,'model',name)
            delay_details=[d[~((d.model==name)&(d.window_size==width))] for d in delay_details]
            delay_details.append(detail)
            result_rows=[r for r in result_rows if (r['model'],int(r['window_size']))!=(name,width)]
            delay_rows=[r for r in delay_rows if (r['model'],int(r['window_size']))!=(name,width)]
            result_rows.append({'model':name,'window_size':width,'first_possible_delay_s':max(width,minimum_index)*.1,
                                **metrics,'threshold':threshold,'validation_mse':validation_mse,
                                'validation_common_mse':common_mse,'family':family,
                                'score_type':params['score_type'],'fit_seconds':fit_seconds})
            delay_rows.append({'model':name,'window_size':width,**delay})
            all_details[label]={'all_timestamps':metrics,'scored_only':scored_metrics,'delay':delay,
                                'threshold':threshold,'training_parameters':params}
            (out_dir/'training'/f'{label}.json').write_text(json.dumps(params,indent=2)+'\n')
            if hasattr(model,'weights'):
                weights=model.weights if isinstance(model.weights,list) else [model.weights]
                np.savez(out_dir/'models'/f'{label}.npz',**{f'weights_{i}':w for i,w in enumerate(weights)})
            if hasattr(model,'estimator'):
                import joblib
                joblib.dump(model.estimator,out_dir/'models'/f'{label}.joblib')
            pd.DataFrame(result_rows).to_csv(out_dir/'timestamp_results.csv',index=False)
            pd.DataFrame(delay_rows).to_csv(out_dir/'delay_results.csv',index=False)
            pd.concat(delay_details).to_csv(out_dir/'detection_delay.csv',index=False)
            (out_dir/'metrics.json').write_text(json.dumps(all_details,ensure_ascii=False,indent=2)+'\n')
            write_report(out_dir,result_rows,delay_rows,planned)
            print(f'seed={seed} {label}: F1={metrics["F1"]:.4f} FP={metrics["FP"]} FN={metrics["FN"]} '
                  f'detected={delay["detected_clips"]}/21 median_delay={delay["median_delay_s"]}',flush=True)
    table=pd.DataFrame(result_rows)
    forecasts=table[table.family=='forecast']
    best_by_val=(forecasts.loc[forecasts.groupby('model').validation_common_mse.idxmin(),['model','window_size']]
                 if not forecasts.empty else forecasts[['model','window_size']])
    selected=set(map(tuple,best_by_val.to_numpy()))
    table['selected_by_normal_validation']=[(r.model,r.window_size) in selected for r in table.itertuples()]
    table.to_csv(out_dir/'timestamp_results.csv',index=False)
    pd.DataFrame(delay_rows).to_csv(out_dir/'delay_results.csv',index=False)
    pd.concat(delay_details).to_csv(out_dir/'detection_delay.csv',index=False)
    (out_dir/'metrics.json').write_text(json.dumps(all_details,ensure_ascii=False,indent=2)+'\n')
    write_report(out_dir,table.to_dict('records'),delay_rows,planned,complete=True)
    return table


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(DEFAULT_DATA))
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS))
    parser.add_argument('--windows',nargs='+',type=int,default=WINDOWS)
    parser.add_argument('--minimum-index',type=int,default=0,
                        help='Fix train/validation/calibration/test target phase across window sizes')
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if any(w<1 or w>=50 for w in args.windows) or args.minimum_index<0 or args.minimum_index>=50:
        parser.error('Windows must be 1..49 and minimum index 0..49')
    WINDOWS=sorted(set(args.windows))
    run(args.data_dir,args.output_dir,args.models,args.minimum_index,args.seed)
