from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';FIG=ROOT/'figures'
d=pd.read_csv(OUT/'seed0_timestamp_diagnostics.csv');s=pd.read_csv(OUT/'seed_metrics.csv')
a=json.loads((OUT/'analysis.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4))
labels=['Warmup: no score','Scored FN before first alarm','Scored FN after first alarm']
cols=['FN_warmup','FN_before_first_alarm','FN_after_first_alarm'];colors=['#8b959e','#df972a','#315779']
left=pd.Series([0,0,0])
for label,col,color in zip(labels,cols,colors):
    ax.barh(['Seed 0','Seed 1','Seed 2'],s[col],left=left,color=color,label=label)
    for i,v in enumerate(s[col]):ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white' if col!='FN_before_first_alarm' else 'black')
    left+=s[col]
ax.set_xlabel('False negative timestamps (600 fault timestamps total)');ax.set_title('No fault clip was completely missed; every seed alerted all 21 clips by 0.5 s')
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=1);fig.tight_layout();fig.savefig(FIG/'fn_breakdown.png',dpi=170,bbox_inches='tight');plt.close(fig)

med=d[d.status=='scored'].groupby('outcome')[[c for c in d.columns if c.endswith('_conditional_percentile')]].median()
fig,ax=plt.subplots(figsize=(8,4))
for name,color in [('FP','#df972a'),('FN','#ba4141'),('TN','#8b959e'),('TP','#315779')]:
    ax.plot(['Upper vibration','Lower vibration','Current'],med.loc[name],marker='o',label=name,color=color)
ax.set(ylim=(0,102),ylabel='Median conditional-score percentile in normal calibration',title='Model-based diagnostic: how rare is each sensor window given the others?')
ax.grid(alpha=.15);ax.legend(frameon=False,ncol=4);fig.tight_layout();fig.savefig(FIG/'conditional_rarity.png',dpi=170);plt.close(fig)

seed_table=['| seed | 오탐 FP | 전체 미탐 FN | 초반 관측 부족 | 첫 경보 전 판정 미탐 | 첫 경보 이후 미탐 | 완전 무경보 이상 클립 |','|---|---:|---:|---:|---:|---:|---:|']
for v in s.itertuples():seed_table.append(f'| {v.seed} | {v.FP} | {v.FN} | {v.FN_warmup} | {v.FN_before_first_alarm} | {v.FN_after_first_alarm} | {v.fault_no_alarm_clips} |')
md='''# 기존 GMM의 남은 오탐·미탐 분석

## 1. 분석 목적과 핵심 결론

**성능이 이미 높은 원래 3센서 GMM이 남긴 소수 오류의 위치와 판정 근거를 확인한다.** 모델·학습 데이터·임계값·윈도우를 변경하지 않았다. 외부 벤치마크가 아닌 KAMP 원래 모델의 사후 오류 분석이다.

1. **완전히 놓친 이상 클립은 없다.** seed0/1/2 모두 이상21클립에0.5초 안에 첫 경보를 냈다. 이 데이터의 클립 시작 기준이다.
2. 전체 시점 미탐82~84개 중42개는 처음2시점의 관측 부족이다. 점수 계산 가능한 미탐은40~42개이며 대부분 첫 경보 이후다.
3. 오탐9~12개는 정상120클립 중 `normal_0570`, `normal_0576` 두 클립에만 몰린다. 세 시드에서 위치가 반복된다.
4. `normal_0570`은 전류 단독의 크기보다 **학습 분포에서 드문 진동 윈도우와 다른 센서와의 조합**이 두드러진다. `normal_0576`의 두 오탐은 큰 전류 윈도우도 함께 드물다. 모든 오탐의 원인을 전류로 묶으면 안 된다.
5. 판정 가능한 미탐은 대체로 **진동이 작고 전류 변화가 상대적으로 완만한 짧은 윈도우**에서 나타난다. 정상 학습 윈도우와 가까워 이상 점수가 임계값 아래에 머문다. 그렇다고 그 시점이 실제 정상이라고 재라벨링하지 않는다.

## 2. 분석한 모델과 재현 확인

- 입력: 상부진동 AI0_Vibration, 하부진동 AI1_Vibration, 전류 AI2_Current.
- 과거2시점+현재의9차원 벡터,10Hz CSV, stride1. 최초 점수는클립 시작0.2초부터.
- 정상599클립: 학습305 / 검증54 / 보정120 / 테스트120. 이상21클립 전체테스트.
- 정상테스트4,090시점, 이상600시점, 총4,690시점.
- GMM 정상 학습BIC로K=1/2/4 선택. 정상 보정점수99% 분위수, strict > 판정. 초기2시점False.
- 저장된 seed0/1/2 모델을 불러왔다. 원본CSV SHA256, 저장된 점수, 재계산 점수, 공유된 판정CSV의 시점 순서와 판정 모두 일치함을 확인했다. 새 모델 학습은 없다.
- 세 seed의 원래F1은0.9172±0.0003. seed는학습초기화 차이며 독립고장 반복이 아니다.

## 3. 시점 미탐과 조기경보 실패를 구분

'''+ '\n'.join(seed_table)+'''

![미탐 분해](figures/fn_breakdown.png)

### A. 초반 관측 부족: 42시점

각 이상클립의index0·1, 즉0.0·0.1초는W=2의 과거 관측이 충분하지 않아 점수 자체가 없다. 이42개는 평가 규격상FN이지만, 점수를 계산하고 정상이라 판단한40~42개와 성격이 다르다. 주평가에서 삭제해F1을 높이지 않고 별도 분해한다.

### B. 첫 경보 전 실제 판정 미탐: 3~5시점

seed0: `fault_0004`의0.2·0.3·0.4초, `fault_0019`의0.2초, `fault_0021`의0.2초다. 첫 경보는각각0.5·0.3·0.3초다. 나머지18클립은 최초 판정 가능한0.2초에경보를 냈다. seed1/2에서는첫 경보 전 미탐3시점이다.

**첫 경보 전 미탐이 조기경보 개선에서 우선 볼 대상**이다. 그러나현재21/21클립의0.5초 경보 목표는달성했다. 더 빠른 목표를 주장하려면0.2/0.3초 경보율과정상오탐의trade-off를평가해야한다.

### C. 첫 경보 이후 미탐: 35~39시점

이미경보를 냈던클립에서짧은구간의점수가 내려가며False가된다. 현재알고리즘은매시점독립임계값판정이며경보상태를유지하지 않는다. 이는시점별F1의오류이지만클립을처음부터끝까지못잡은사례는아니다.

경보를유지하는운영정책은알림의깜빡임을줄일수있지만, 그정책으로과거/이후의모든시점을True로바꿔모델F1이개선됐다고말하면안된다. 센서판정과설비알림상태를분리해서평가한다.

## 4. 오탐은 어떤 조건인가?

### normal_0570: 드문 진동 및 센서 조합

seed0에서10FP, seed1/2에서8FP. seed0의오탐구간은0.2초(1점),0.6~0.7초(2점),1.0~1.3초(4점),2.0~2.2초(3점)다. 한시점만튀는오류로묶을수없다.

정상라벨클립이지만원래정상학습분포에서드문진동을보인다. 조건부진동점수는오탐10점모두상부진동에서정상보정99백분위보다높다. 같은시점의전류조건부점수는모두99백분위보다낮다. 따라서이클립의오탐을‘전류가커서’라고만설명할수없다.

모델은현재값하나가아닌최근3시점을보므로, 현재진동값이작아져도직전큰진동을포함한윈도우는높은점수를낼수있다.

![오탐 클립](figures/normal_0570.png)

### normal_0576: 드문 전류 윈도우를 포함한 조합

seed0의2.0·2.1초에서2FP, seed1/2는2.0초의1FP. 현재전류는각각261.15·258.72, 전류조건부점수는정상보정99.57·99.60백분위다. seed0의임계값초과폭은약1.20·0.20으로첫클립의최대초과폭14.54보다작다. 임계값주변의경계판정과드문전류/진동조합사례로본다.

![두 번째 오탐 클립](figures/normal_0576.png)

### 두 클립의 해석 범위

데이터제공라벨기준으로FP다. 드문정상운전패턴인지, 수집조건변화인지, 라벨이지시하는상태와센서의순간변화가다른지알수없다. 분석만으로실제고장이었다고뒤집거나라벨오류로확정하지않는다. 설비부하·작업조건·정비기록이있으면확인할대상이다.

## 5. 점수를 계산하고도 놓친 시점의 특징

아래는seed0의점수계산가능시점만비교한중앙값이다. 수치는조건을설명하기위한표본요약이며인과효과가아니다. 센서단위는원본의정의가불명확하므로임의물리단위를붙이지않는다.

| 특징 | 미탐 FN (40점) | 탐지 TP (518점) | 정상 무경보 TN (3,839점) |
|---|---:|---:|---:|
| 최근3시점 상부진동 최대절댓값 | 0.069 | 0.205 | 0.059 |
| 최근3시점 하부진동 최대절댓값 | 0.087 | 0.199 | 0.065 |
| 현재 전류 1시점 변화절댓값 | 27.42 | 57.22 | 31.45 |
| 최근3시점 전류 최대절댓값 | 135.90 | 141.86 | 111.24 |
| 가장 가까운 정상학습 윈도우 거리 | 0.924 | 2.302 | 0.563 |
| 점수−임계값 | −4.555 | 127.298 | −9.018 |

거리는정상학습표준화9차원공간에서의Euclidean거리다. 진동/전류의절대단위가아닌학습표준편차단위다. 거리그자체를새판정규칙으로사용하지않았다.

**전류절대크기만으로FN/TP차이를설명하기어렵다.** 미탐에서는진동이작고전류의국소변화도작아져전체윈도우가정상패턴과더가까워진다. 개별채널조건부희귀도중앙값도FN은약82~85백분위,TP는모두100백분위다.

40FN중임계값바로아래1점수이내는1개뿐,3점수이내는11개다. 따라서작은임계값조정만으로남은미탐을모두해결할수있다는근거는없다. 임계값을낮추면정상오경보가얼마나증가하는지도함께봐야한다.

### 대표 사례: fault_0021

진동전체가작은클립이다.0.2초윈도우점수5.42는임계값15.00보다훨씬낮고0.3초에처음경보가난다. 이후전류변화가완만해지는0.9·1.0·1.2·1.3초에서다시False가된다. 클립전체를놓친사례가아니라정상과겹치는짧은패턴에서판정이내려오는사례다.

![미탐 클립](figures/fault_0021.png)

### 다른 대표 사례

- `fault_0004`: 진동이상대적으로작고0.2~0.4초를놓쳐첫경보0.5초. **초기경보지연이남는대표사례**다.
- `fault_0016`, `fault_0017`: 판정미탐7·6점으로많지만둘다첫경보는0.2초다. 뒤의조용한시점에서점수가내려오는지그래프로확인한다.
- 전체12이상클립에판정미탐이분산됐다. 가장많은클립하나만으로전체원인을일반화하지않는다.

![초기 경보 지연 사례](figures/fault_0004.png)

## 6. 모델이 무엇을 드물다고 보는지 확인한 방법

조건부점수는학습된GMM에서 `−log p(한 센서의 3시점 | 다른 두 센서의 3시점)`을계산했다. 완전한9차원우도와해당센서를제외한6차원주변우도차이로정확히계산한다. 각센서의점수를해당정상보정점수분포의백분위로변환해비교했다.

이진단은모델내에서센서패턴의희귀도를설명한다. 세조건부점수는원래점수의독립적인가산기여분이아니며,센서고장/물리적원인을증명하는인과기여도가아니다. 새모델·새임계값으로사용하지않았다.

![조건부 희귀도](figures/conditional_rarity.png)

FP의조건부백분위중앙값: 상부진동99.91,하부진동99.06,전류71.37. 이는첫오탐클립이진동관련패턴을포함한다는설명을지지한다. 단12점과두클립의표본이므로현장전체오탐원인으로확대하지않는다.

`training_quantile_conditions_seed0.csv`는정상학습센서절댓값p50/p90/p99로구간을미리정해오류율을집계했다. 상부진동p99초과정상관측의FP는2/7이지만모수가매우작다. ‘이조건이면28.6%오탐’이라는일반화주장보다오류가몰리는조건을찾는참고로사용한다.

## 7. 시드에 따라 바뀌는가?

- 세시드가모두FP로판정한정상시점9개. 어느시드라도FP인시점12개. 오탐클립은항상같은두개다.
- 세시드가모두놓친점수계산가능이상시점38개. 초기화변경만으로사라지지않는오류가대부분이다.
- 세시드모두21/21이상클립을0.5초이내경보했다.
- threshold는seed0=15.0028, seed1=15.5221, seed2=15.4164. 모델우도가각기달라숫자만으로seed별임계값을직접맞추지않는다.

## 8. 개선 우선순위 — 아직 실험하지 않은 제안

### 1순위: 드문 정상 진동/변수 조합의 오경보

정상검증·보정에서도유사한희귀운전패턴이있는지확인하고학습데이터가정상상태의다양성을충분히포함하는지검토한다. 새로운정상기록의확보또는정상상태별분포가후보다. **테스트오탐두클립을학습에추가한뒤같은테스트에재평가하지않는다.**

연속2~3회경보등정책은한시점FP를줄일수있지만첫클립에는2·3·4점연속FP가있어모두해결되지않는다. 탐지지연과같이평가해야한다. 이번에는정책값을테스트에맞춰고르지않았다.

### 2순위: 첫 경보 전 미탐 3~5시점

`fault_0004`등에서진동이작아도전류와진동의관계변화가더빨리드러나는지후속연구로볼수있다. 윈도우관측부족42점은짧은입력용별도모델의과제이며원모델임계값으로해결되지않는다.

### 3순위: 첫 경보 후 False로 돌아오는 운영 문제

알림상태유지·히스테리시스는운영단에서검토할수있다. 실제정상복귀시점정답이없어현재데이터로얼마동안유지할지확정하기어렵다. 시점별모델F1과알림정책지표를분리한다.

## 9. 발표용 결론

“세시드모두21개이상클립에0.5초이내경보했으나일부시점에서미탐이남았습니다. 이를분해한결과42시점은초기관측부족이었고,판정미탐의대부분은이미첫경보를낸뒤센서패턴이정상분포와겹치는구간에서발생했습니다. 오경보는같은두정상클립에집중됐으며,특히진동과센서간조합의희귀성이두드러졌습니다. 따라서개선대상을드문정상운전패턴의포괄성과초기약한패턴의탐지로구체화했습니다.”

## 10. 한계와 파일

- 이상21클립은한기록의구간이며21독립고장이아니다. 실제고장발생시점과시점별고장가시성정답이없다.
- 정상/이상수집날짜차이·전류기록조건교란가능성은이번분석으로해소되지않는다. 좋은오류분해가현장일반화증명은아니다.
- 테스트라벨을보고한사후설명이다. 이후이사례로모델을개선하면같은테스트성능은개발결과이며독립검증이필요하다.
- 그림대표는seed0이며전체오류시점CSV·세시드합의표를보존해유리한사례만선택하지않았다.
- `results/seed_metrics.csv`: 세시드오류분해.
- `results/all_error_points.csv`: 모든시드의FP/FN,원시값·점수·진단지표. 로컬분석용이며원자료를포함하므로Git자동업로드하지않았다.
- `results/error_runs.csv`: 연속오류구간.
- `results/clip_summary_seed0.csv`: 모든141테스트클립요약.
- `results/seed_consensus.csv`: 세시드경보합의.
- `results/nearest_correct_pairs_seed0.csv`: 같은정답라벨의잘판정된최근접사례와비교. 관찰상유사성이지동일조건실험이아니다.
- `results/outcome_features_seed0.csv`: FP/FN/TP/TN특징요약.
- `results/analysis.json`: 해시·정량결과.
- `figures/`: 원시3센서+점수+임계값에오류를표시한클립6개,오류분해/조건부희귀도그림.

원래저장된모델/표준화설정/판정CSV와원본데이터를유지한상태에서다음명령으로재현한다.

```bash
python code/analyze.py --experiment-root /path/to/press_forecast_experiment --code-dir /path/to/team_shared/gmm_audit/code --data-dir /path/to/original_csv_folder
python code/build_report.py
```

실행환경은기존Python3.12+NumPy2.5.3+pandas2.2.3+scikit-learn1.9.1이며SciPy/joblib/matplotlib을사용했다. 임계값/모델파일을불러오므로새로학습하는재현과구별된다.
'''
(ROOT/'GMM_ERROR_ANALYSIS.md').write_text(md)
print('Report saved:',ROOT/'GMM_ERROR_ANALYSIS.md')
