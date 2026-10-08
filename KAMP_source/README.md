# KAMP 유압펌프 조기 이상탐지 소스코드 제출

W=3 통일 보고서와 발표자료의 최종 모델·평가·부록 실험과 대응하는 통일본.

**제출 모델: 정상 데이터로 학습한 GMM + 초기 관측 주변분포 판정(StartupMarginal, point).** 상부·하부 진동과 전류를 이용해 매 시점 정상/이상을 판단한다. 미래 관측은 사용하지 않는다.

## 1. 핵심 파일

```text
code/final_model_pipeline.py          전처리부터 최종 결과까지 한 번에 실행
code/recommended_detector.py          센서값을 한 시점씩 받는 최종 판정기
train.py / predict.py / evaluate.py   개별 학습·예측·평가
src/                                 데이터 분할, GMM 주변분포, 순차 판정
requirements.txt                     최종 모델 실행 환경
models/                              시드 0·1·2 저장 모델·정규화·임계값
predictions/test_predictions.csv      기본 제출 예측 결과 (시드 0)
predictions/seeds/                    시드 1·2 및 기존 GMM 비교 예측
results/                             최종 평가, 평균 ± 표준편차, 첫 경보 시간
data/                                학습·검증·보정·테스트 CSV, 클립 분할
  raw/                               제공된 원본 정상·이상 CSV
comparisons/baselines/                모든 베이스라인의 통합 실행 코드·결과
comparisons/improvements/             오류 진단·개선 후보 96회 실험
comparisons/report_results/           보고서의 W·센서·전류 통제 비교 결과
```

## 2. 실행

Python **3.12**, CPU 기준. 압축 해제한 `KAMP_source` 폴더에서 실행한다.

```bash
python -m pip install -r requirements.txt
# 포함된 저장 모델로 테스트 예측 → 평가
python predict.py
python evaluate.py --predictions runs/test_predictions.csv
# 보고서와 같은 전처리 → 학습 → 보정 → 시드 0·1·2 예측·평가
python code/final_model_pipeline.py --data-dir data --output-dir runs/final_model
```

기본 출력은 `runs/`에 저장되며 제출 결과 파일은 보존된다. `--seed 1` 또는 `2`로 다른 시드의 예측이 가능하다. 기본 W=3 GMM(초반 세 시점 미판정)은 `python predict.py --mode original --output runs/original.csv`로 실행한다. 원본에서 분할 파일을 다시 만들려면 `python prepare_data.py`를 실행한다. 보고서 제6장의 `code/` 경로로도 실행할 수 있다. 베이스라인 구현은 `comparisons/baselines/` 한 폴더에 모았고 `code/`의 해당 파일은 이를 호출하는 진입점이다.

```bash
# 전체 비교·신경망 환경 (최종 GMM만 실행하면 설치 불필요)
python -m pip install -r code/requirements.txt
# 기본 GMM 센서 제거 실험
python code/audit_current.py --data-dir data --output-dir runs/basic_sensor
# 최종 GMM 센서 제거·W 비교·전류 통제 및 8개 기존 탐지기 비교
python sensor_shortcut_check.py --data-dir data --output-dir runs/report_checks
```

전체 17개 방법은 `code/run_experiment.py`에서 실행한다. W 범위 1·2·3·5·8·10·15·20·30과 시드 0·1·2 실행 예시는 `comparisons/baselines/README.md`를 따른다. 최종 제출 설정은 **W=3**이다. W 비교 결과를 참고해 변경했으며 독립 검증 전의 탐색적 설정이다. 개선 후보 96회 실험은 `comparisons/improvements/README.md`를 따른다.

## 3. 데이터 및 학습 규칙

출처: KAMP **소성가공 예지보전 AI 데이터셋**. 원본 SHA256은 `data/provenance.json`에 있다. 동일 시각의 중복 행을 제거하고, 인접 관측 간격이 **0.5초 초과**이면 새 클립으로 나눈다. 클립 내부 관측은 약 0.1초 간격이다. 경계를 넘는 보간이나 윈도우를 사용하지 않는다.

| 용도 | 정상 클립 | 이상 클립 | 역할 |
|---|---:|---:|---|
| 학습 | 305 | 0 | 정규화 및 GMM 추정 |
| 검증 | 54 | 0 | 기존 방법 비교; 최종 GMM 추정에는 미사용 |
| 보정 | 120 | 0 | 경보 임계값 계산 |
| 테스트 | 120 | 21 | 4,090 정상 + 600 이상 시점 평가 |

정상은 시간순으로 분리하며 동일 클립이 여러 분할에 섞이지 않는다. 학습·검증·보정·테스트의 클립 목록은 `data/clip_splits.csv`에 있다. `train.py`는 `train.csv`와 `calibration.csv`만 읽는다. 테스트 라벨은 추정이나 임계값 계산에 사용하지 않는다.

센서 순서: `AI0_Vibration`(상부 진동), `AI1_Vibration`(하부 진동), `AI2_Current`(전류). 단위는 원본 제공값을 따른다. `data/*.csv`의 `clip_id`, `sample_index`, `timestamp`, `elapsed_seconds`로 원본 시점과 연결한다. `source_row`는 원본 CSV의 첫 번째 열 값이다.

## 4. 모델과 판정

정상 학습값의 평균·모표준편차로 정규화한다. 과거 3개+현재 1개 시점의 3센서(12차원)에 full-covariance GMM을 적합한다. K∈{1,2,4} 중 정상 학습 BIC 최소를 선택한다. `reg_covar=1e-4`, `n_init=2`, `max_iter=200`, 시드 **0·1·2**를 사용한다.

클립 최초 1/2/3개 관측은 저장된 12차원 GMM의 **3/6/9차원 주변분포**, 이후는 최근 4개 관측의 12차원 분포를 사용한다. 점수는 음의 로그 밀도이며 클립 시작마다 상태를 초기화한다. 초기 1/2/3개 위치는 정상 보정 클립의 같은 위치 점수의 99백분위, 이후는 정상 보정의 완전한 창 점수의 99백분위로 임계값을 정한다. **점수 > 임계값**이면 이상이다. stride=1이며 경보 유지·연속 판정·정답 기반 소급 보정은 적용하지 않는다.

## 5. 예측 파일 및 성능

`predictions/test_predictions.csv`는 시드 0의 **4,690개 모든 테스트 시점**에 대한 결과다. 주요 열: `clip_id`, `sample_index`, `timestamp`, `elapsed_seconds`, `y_true`(정상 0/이상 1), `pred`(True/False), `score`, `threshold`, `dimension`, `seed`, `mode`. `pred`가 최종 제출 판정이다. 시드별 파일은 표준편차 확인용이며 앙상블이 아니다. 라벨 없이도 예측할 수 있다.

시드 0·1·2 평균 ± 표본 표준편차: F1 **0.9650 ± 0.0013**, 오탐 **11.67개**, 미탐 **29.67개**, 평균 첫 경보 **0.108초**, 0.5초 이내 경보 **21/21클립**. 기본 W=3 GMM은 F1 0.9278, 평균 첫 경보 0.305초(경보한 20클립 기준)이다. 초기 판정 보완으로 평균 FN 71→29.67개, FP 11.33→11.67개가 되었다. 정확한 시드별 수치는 `results/metrics_by_seed.csv`, `summary.csv`를 따른다.

F1=2TP/(2TP+FP+FN), 오탐률=FP/(FP+TN), 미탐률=FN/(TP+FN). 첫 경보 지연은 **클립 시작부터 첫 True까지**이며, 미경보 클립은 별도로 집계한다. 준비 시점도 전체 평가에 포함한다. 첫 관측의 0.0초 판정은 관측 수신 시점의 경보이며 실제 장치 지연 0초라는 의미는 아니다.

## 6. 해석 범위

21개 이상 클립은 하나의 이상 기록에서 나온 구간이며 독립적인 21개 고장 사례가 아니다. 파일/클립 상태 라벨을 시점별 평가에 사용하므로 정밀한 고장 시작 시각은 알 수 없다. 정상·이상 수집 날짜와 전류 기록 조건의 교란 가능성이 남아 있다. 후보 설계 및 선택과 W=3 변경에서 테스트 결과를 참고했으므로 **탐색적 개발 결과이며 독립 현장 검증 성능이 아니다.** 설비 점검을 돕는 경보 모델로 제시하며 실제 자동 정지 제어의 검증은 별도다.
