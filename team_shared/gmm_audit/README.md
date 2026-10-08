# GMM 센서 기여·전류 기록 조건 검토

`team_shared/gmm_audit/`에서 공유 자료를 관리한다. 팀원 저장소 `wodtod68-sys/sosunggongmo_KCC`의 기존 실험은 NBM-T² 클립 특징 모델이며, 이 폴더는 시점별GMM 파이프라인이다. 분할·윈도우·임계값이 다르므로 두 실험의 점수를 같은 규격 결과로 혼합하지 않는다.

## 먼저 읽을 파일

- `docs/CURRENT_SENSOR_AUDIT.md`: 최신 결론, 센서 제거 결과, 확인한 사실과 미확인 원인.
- `docs/WINDOW_ANALYSIS.md`: 전체 파이프라인과 기존 탐색 기록. 1~20절은3센서 역사적 결과,21절이 최신 해석.
- `presentation/유압펌프_GMM_센서검토_발표자료.pptx`: 최신 편집 가능한 발표자료.
- `results/sensor_audit/summary.csv`: GMM W=2 센서별평균±표본SD.
- `results/seed_summary.csv`, `seed_runs.csv`: 기존17방법×9W×2조건,522실행. 모두3센서 결과.

## GMM 센서 검토 재현

원본 CSV는 저장소에 포함하지 않는다. 로컬의 같은 폴더에 `press_data_normal.csv`, `outlier_data.csv`를 준비한다. SHA256은 `results/sensor_audit/audit.json`을 확인한다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements-audit.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/audit_current.py --data-dir /path/to/data --output-dir local_results/sensor_audit
```

저장된 검토 실행환경: Python3.12.14, NumPy2.5.3, pandas2.2.3, scikit-learn1.9.1. 설치버전이 다르면 같은 수치의 재현을 보장하지 않는다. 실행하면9개GMM 결과, 판정CSV, 집계표와 진동 모델이 로컬 생성된다. CSV판정에는 원시센서값을 포함하지 않는다.

## 기존 3센서 전체 실험

전체 실행에는PyTorch도 필요하다. `code/requirements.txt` 설치 후 아래와 같이 각seed를 별도 출력 폴더로 실행한다. 고정방법은seed0에서만 실행해도 된다. 다른seed의 고정방법을 다시 돌리는 경우 결과를3회 확률모델 반복으로 집계하지 않는다.

```bash
python code/run_experiment.py --data-dir /path/to/data --output-dir local_results/natural_seed0 --windows 1 2 3 5 8 10 15 20 30 --seed 0
python code/run_experiment.py --data-dir /path/to/data --output-dir local_results/natural_seed1 --windows 1 2 3 5 8 10 15 20 30 --seed 1 --models GMM MLP32 LSTM32 GRU32 AE_Recon IsolationForest
python code/run_experiment.py --data-dir /path/to/data --output-dir local_results/natural_seed2 --windows 1 2 3 5 8 10 15 20 30 --seed 2 --models GMM MLP32 LSTM32 GRU32 AE_Recon IsolationForest
```

학습부터 통제하는 C조건은같은명령에 `--minimum-index 30`을 추가하고 별도 출력 폴더를 사용한다. B조건은A모델의 판정에서index≥30만 집계한다. 이들 후반평가는 실제조기경보의 대체지표가 아니다.

원본데이터·대용량모델·가상환경·개인경로·임시 렌더링은공유하지 않는다. Git에 들어간별도판정CSV는 재현 확인용테스트라벨/시점/판정으로만 구성된다.
