# GMM 오탐·미탐 분석 및 초기 판정 개선

2026-10-08 팀 공유본. 기존 전류 기록 조건 감사는 `../gmm_audit/`에, 이번 오류 분석과 개선 실험은 이 폴더에 구분했다.

## 먼저 읽을 파일

**[GMM_ERROR_ANALYSIS.md](GMM_ERROR_ANALYSIS.md)**: 문제 정의, 데이터 분할, GMM 이론과 적용, 오류 조건, 96회 개선 실험, 최종 권고, 한계 및 발표 구성. 보고서·PPT 작성은 이 문서를 기준으로 한다.

## 핵심 결과

시드 0·1·2의 평균: 초기 주변분포 판정으로 F1 **0.9172 → 0.9388**, 평균 첫 경보 **0.217초 → 0.108초**. 오탐은 평균 **10개로 동일**, 미탐은 **83.33 → 60.33개**. 0.5초 이내 경보 21/21, p90과 0.2초 경보율은 동일하다. 초기 시점부터 관측된 값만 사용하며 미래값을 넣지 않는다.

후보 설계 및 선택에서 기존 테스트 오류를 참고한 **탐색적 개발 결과**다. 독립 현장 검증 성능으로 해석하지 않는다. 정상·이상 수집 날짜 및 전류 기록 조건의 교란 가능성은 계속 남아 있다.

## 파일 구성

- `code/`: 오류 분석, 개선 실험, 보고서 생성, 한 시점씩 실행하는 최종 판정기.
- `figures/`: 오탐·미탐 예시와 개선 결과 그림 9개.
- `results/`: 오류 조건 및 시드별 집계.
- `results/improvements/runs.csv`: 8후보 × 4경보정책 × 3시드 = 96실행.
- `results/improvements/summary.csv`: 평균 ± 표본 표준편차.
- `results/improvements/protocol.json`, `selection.json`: 실험 정의와 선택 기준.
- `results/improvements/predictions_96_runs.zip`: 96개 시점별 판정 CSV. 압축을 같은 폴더에 풀면 실험 출력 구조가 복원된다. 원시 센서값은 포함하지 않는다.
- `results/improvements/recommended/`: 최종 권고 모델과 표준화·임계값 설정, 시드 0·1·2 각각.

## 최종 모델 실행

Python 3.12에서 이 폴더를 현재 디렉터리로 사용한다.

```bash
python -m pip install -r code/requirements.txt
```

```python
import sys
sys.path.insert(0, 'code')
from recommended_detector import StartupMarginalGMM
model = StartupMarginalGMM('results/improvements/recommended', seed=0)
model.reset()  # 새 클립의 시작마다 호출
# upper, lower, current는 동일 시점의 실제 센서값
result = model.update([upper, lower, current])
print(result)  # anomaly, score, threshold, observed_points, dimension
```

센서 순서: 상부 진동 `AI0_Vibration`, 하부 진동 `AI1_Vibration`, 전류 `AI2_Current`. 최초 1/2/3개 관측에서 3/6/9차원 주변분포를 평가하며 이후 최근 3시점만 사용한다. 시드 0은 실행 예시이며 최고 시드를 골라 배포한 것이 아니다.

## 전체 분석 재현

원본 KAMP CSV와 기존 실험의 저장 모델·설정·예측 결과가 별도로 필요하다. 이번 공유본은 원시 CSV나 기존 전체 실험 폴더를 포함하지 않는다. 아래 `EXPERIMENT_ROOT`는 기존 `press_forecast_experiment` 폴더를 가리키도록 지정한다.

필수 기존 경로:

- `window_sweep/natural/results/` (seed 0)
- `seed_experiments/natural/seed_1/results/`, `seed_2/results/`
- 각 결과 폴더의 `experiment_config.json`, `models/GMM_W2.joblib` 및 기존 판정 CSV

원래 실험 코드는 [../gmm_audit/code/](../gmm_audit/code/)에 있다. 저장된 결과로 오류를 재현하는 절차와 처음부터 재학습하는 절차는 구별한다.

```bash
python code/analyze.py --experiment-root "$EXPERIMENT_ROOT" --code-dir ../gmm_audit/code --data-dir "$DATA_DIR"
python code/build_report.py
python code/improve.py --experiment-root "$EXPERIMENT_ROOT" --code-dir ../gmm_audit/code --data-dir "$DATA_DIR"
python code/finalize_report.py
```

`finalize_report.py`가 마지막 단계다. `build_report.py`만 실행하면 개선 실험 이전 보고서가 생성된다. 최종 실행 모델은 위 재현 입력 없이도 포함된 모델과 설정으로 사용할 수 있다.
