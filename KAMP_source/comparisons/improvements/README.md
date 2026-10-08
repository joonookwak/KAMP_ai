# 개선 후보 96회 실험

8개 점수 구성 × 4개 경보 정책 × 시드 0·1·2. 저장된 원래 GMM을 사용하며 정상 보정 데이터로 임계값을 계산한다.

루트에서 실행:

```bash
python comparisons/improvements/code/improve.py --experiment-root comparisons/improvements/original_inputs --code-dir comparisons/baselines --data-dir data/raw
```

현재 `results/`에 집계 결과가 있다. 전체 시점별 판정은 `predictions_96_runs.zip`에 묶었다. 재실행하면 `results/`에 96개 CSV가 생성된다. 모델·설정·원래 점수 검증 입력을 `original_inputs/`에 포함했으므로 외부 로컬 경로가 필요 없다. 정상 학습/보정만으로 추정하지만 후보 설계와 선택에서 테스트 오류를 참고한 탐색적 비교다. 기본 최종 모델은 StartupMarginal + point이며 히스테리시스 결과와 혼동하지 않는다.
