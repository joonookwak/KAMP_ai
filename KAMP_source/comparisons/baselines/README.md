# 베이스라인 통합 실행

모든 비교 방법을 `run_experiment.py` 한 실행기로 묶었다. 예측 계열은 `models.py` 및 `recurrent_models.py`, 재구축·거리·밀도 계열은 `ad_models.py`에 있다. 최종 제출 모델은 루트의 `train.py` / `predict.py`이다.

루트 디렉터리에서 실행:

```bash
python -m pip install -r comparisons/baselines/requirements.txt
python comparisons/baselines/run_experiment.py --data-dir data/raw --output-dir runs/baselines/seed0 --models GMM Mahalanobis PCA_Recon AR AR_Relation --windows 2 --seed 0
```

`--models`를 생략하면 전 방법을 실행한다. LSTM32·GRU32·AE_Recon에는 PyTorch가 필요하다. `--windows 1 2 3 5 8 10 15 20 30`으로 전체 W 비교가 가능하며, 동일 평가 가능 시점 통제는 `--minimum-index 30`이다. 확률적 방법(MLP32, LSTM32, GRU32, AE_Recon, GMM, IsolationForest)은 `--seed 0`, `1`, `2`로 각각 실행한다. 나머지는 기존 실험에서 1회 실행했다.

정상 학습 305 / 검증 54 / 보정 120 / 테스트 120 클립, 이상 21클립은 테스트 전용이다. 원래 비교 방법은 W 준비 구간의 경보를 False로 두며 모든 타임스탬프를 평가한다. 최종 모델의 초기 주변분포 개선과 구별한다. `results/seed_runs.csv`와 `seed_summary.csv`는 기존 전체 W·시드 실험의 보존 결과다. `natural`과 `controlled` 평가를 섞지 않는다.
