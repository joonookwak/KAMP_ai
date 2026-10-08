# 보고서 표와 결과 파일 대응

- 제2장 표3: `report_table3_baselines_W2.csv`, 루트 `results/summary.csv`의 최종 모델.
- 제2장 표5·제3장 표7: 루트 `results/metrics_by_seed.csv`, `summary.csv`; 기본 오류 분해는 `../improvements/results/seed_metrics.csv`.
- 제3장 표6·부록 표13/14: `GMM_summary.csv`의 condition=original, input별/window별 값. `mode=original`은 기본, `startup`은 최종이다.
- 부록 표15: `recording_baselines_summary.csv`의 8방법; 최종 GMM은 `GMM_summary.csv`의 condition별 값.
- GMM 원본 0.938823, 정상 전류 반올림 0.938516, 이상 전류 격자 제거 0.938823, 레벨 제거 0.902417. 보고서는 소수점 3자리로 반올림한다.

조건 이름: original=원본, normal_rounded=정상 전류 간격 반올림, fault_jittered=이상 전류 ±1.1920929/2 균등 잡음, level_removed=창 내 전류 차분만 사용. 확률적 방법은 시드 0·1·2 평균과 표본 표준편차이며 독립 고장 표본에 대한 신뢰구간이 아니다. 추가 실험도 테스트 오류를 참고한 탐색적 개발 비교다.

재현 명령은 루트 README의 `sensor_shortcut_check.py`를 사용한다. 저장 모델 없이 정상 데이터에서 재학습한다. 원래 모델이 저장된 실험 폴더가 있다면 `comparisons/report_checks.py --experiment-root 경로`로 GMM W 비교에 재사용할 수 있다. 생성되는 수치는 기본/최종을 같은 분할·같은 가우시안 모델로 비교한다.
