# 🚀 AutoML System 빠른 시작 가이드

## 1단계: 환경 설정

### Python 환경 확인
```bash
python --version  # Python 3.7 이상 필요
```

### 패키지 설치
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# 또는 전체 기능 설치
pip install -e ".[full]"
```

## 2단계: 시스템 테스트

```bash
# 시스템 정상 작동 확인
python test_system.py
```

✅ 모든 테스트가 통과하면 다음 단계로 진행하세요.

## 3단계: 예시 데이터 생성

```bash
# 예시 데이터 생성
python example_usage.py
```

이 명령어로 `example_data/` 폴더에 다음 파일들이 생성됩니다:
- `classification_example.csv` (분류용)
- `regression_example.csv` (회귀용)  
- `imbalanced_classification.csv` (불균형 분류용)
- `mixed_data.xlsx` (Excel 형식)

## 4단계: 첫 번째 실행

### 단일 파일 처리
```bash
python main.py --single_file example_data/classification_example.csv --output_folder results/first_test
```

### 배치 처리 (폴더 전체)
```bash
python main.py --input_folder example_data --output_folder results/batch_test
```

## 5단계: 결과 확인

처리가 완료되면 출력 폴더에서 다음을 확인할 수 있습니다:

```
results/
├── summary.json          # 요약 결과
├── results.yaml          # 상세 결과
├── visualizations/       # 시각화
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── shap_analysis.png
└── logs/                 # 처리 로그
```

### 주요 결과 해석

**summary.json 예시:**
```json
{
  "task_type": "classification",
  "best_model": "xgb",
  "best_synthesis": "smote", 
  "performance": 0.8542,
  "data_shape": [200, 5]
}
```

- `task_type`: 자동 감지된 태스크 유형
- `best_model`: 최적 모델
- `best_synthesis`: 최적 데이터 합성 방법
- `performance`: 교차 검증 성능 점수

## 🎯 실제 데이터 사용하기

### 데이터 준비 요구사항

1. **지원 형식**: CSV, Excel, JSON, Parquet
2. **타겟 컬럼**: 다음 중 하나의 이름 사용
   - `target`, `label`, `class`, `y`, `output`, `result`
   - 또는 마지막 컬럼이 자동으로 타겟으로 인식됨

### 예시 데이터 구조

**분류용 CSV:**
```csv
feature_1,feature_2,feature_3,target
1.2,3.4,5.6,0
2.1,4.3,6.5,1
...
```

**회귀용 CSV:**
```csv
size,age,location,price
100,5,A,250000
150,10,B,350000
...
```

## ⚙️ 고급 설정

### 설정 파일 사용
```bash
python main.py --single_file data.csv --output_folder results --config config.yaml
```

### 주요 설정 옵션 (config.yaml)

```yaml
# 최적화 횟수 조정 (속도 vs 성능)
optimization_trials: 50    # 더 많을수록 좋은 결과, 더 오래 걸림

# 사용할 모델 선택
models:
  classification: ['rf', 'xgb', 'lgb']  # 빠른 모델들만
  regression: ['rf', 'xgb', 'lgb']

# 데이터 합성 방법 선택  
synthesis_methods: ['smote', 'adasyn']  # 빠른 방법들만

# 병렬 처리
performance:
  n_jobs: 4  # CPU 코어 수에 맞게 조정
```

## 🔧 문제 해결

### 자주 발생하는 문제들

1. **패키지 import 오류**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **메모리 부족**
   - config.yaml에서 배치 크기 줄이기
   - `n_jobs: 1`로 설정

3. **타겟 컬럼 인식 실패**
   - 컬럼명을 'target'으로 변경
   - 타겟 컬럼을 마지막에 배치

4. **처리 시간이 너무 오래 걸림**
   - `optimization_trials: 10`으로 줄이기
   - 빠른 모델만 사용: `['rf', 'lgb']`

### 성능 향상 팁

1. **소량 데이터 (<1000 샘플)**
   - SMOTE, ADASYN 합성 방법 권장
   - Random Forest, LightGBM 모델 권장

2. **불균형 데이터**
   - BorderlineSMOTE, ADASYN 사용
   - F1-score로 평가

3. **고차원 데이터 (특성 > 50개)**
   - 자동 특성 선택 활성화됨
   - PCA 고려

## 📞 도움이 필요하시면

1. **로그 파일 확인**: `logs/automl.log`
2. **테스트 실행**: `python test_system.py`
3. **예시 실행**: `python example_usage.py`

---

🎉 **축하합니다!** 이제 AutoML 시스템을 사용할 준비가 되었습니다!