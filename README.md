# AutoML System for Small Datasets

소량의 데이터를 위한 자동 머신러닝 시스템입니다. 분류/회귀 태스크를 자동으로 인식하고 다양한 데이터 합성 방법을 통해 최적의 모델을 찾아줍니다.

## 🌟 주요 기능

- **자동 태스크 인식**: 데이터를 분석하여 분류/회귀 태스크를 자동으로 판별
- **다양한 데이터 합성**: SMOTE, ADASYN, CTGAN, Copulas 등 여러 합성 방법 자동 테스트
- **하이퍼파라미터 최적화**: Optuna를 이용한 자동 하이퍼파라미터 튜닝
- **배치 처리 지원**: 폴더 단위로 여러 파일을 한번에 처리
- **프로덕션 레벨**: 안정적이고 확장 가능한 아키텍처

## 📦 설치

```bash
# 저장소 클론
git clone <repository-url>
cd auto-ml-system

# 의존성 설치
pip install -r requirements.txt
```

## 🚀 사용법

### 배치 처리 (폴더 단위)

```bash
python main.py --input_folder ./data --output_folder ./results
```

### 단일 파일 처리

```bash
python main.py --single_file ./data/sample.csv --output_folder ./results
```

### 설정 파일 사용

```bash
python main.py --input_folder ./data --output_folder ./results --config config.yaml
```

## 📁 지원 파일 형식

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)

## 🛠️ 시스템 구성

```
auto_ml_system/
├── core.py              # 메인 파이프라인
├── data_processor.py    # 데이터 전처리
├── task_detector.py     # 태스크 유형 감지
├── data_synthesizer.py  # 데이터 합성
├── model_trainer.py     # 모델 훈련
├── evaluator.py         # 모델 평가
└── utils.py            # 유틸리티 함수
```

## 🔧 주요 컴포넌트

### 1. 데이터 전처리 (DataProcessor)
- 다양한 파일 형식 자동 로딩
- 결측치 처리
- 범주형 변수 인코딩
- 특성 스케일링 및 선택

### 2. 태스크 감지 (TaskDetector)
- 분류/회귀 자동 판별
- 데이터 복잡도 분석
- 클래스 불균형 감지
- 맞춤형 추천 제공

### 3. 데이터 합성 (DataSynthesizer)
- SMOTE, ADASYN, BorderlineSMOTE
- CTGAN (생성적 적대 신경망)
- Gaussian Copula
- 성능 기반 자동 선택

### 4. 모델 훈련 (ModelTrainer)
- Random Forest, XGBoost, LightGBM, CatBoost, SVM
- Optuna 하이퍼파라미터 최적화
- 교차 검증 기반 평가

### 5. 모델 평가 (ModelEvaluator)
- 다양한 평가 지표
- 시각화 레포트 생성
- SHAP 기반 모델 해석

## 📊 출력 결과

처리 완료 후 다음과 같은 결과가 생성됩니다:

```
output_folder/
├── results.yaml          # 상세 결과
├── summary.json          # 요약 정보
├── visualizations/       # 시각화 결과
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── shap_analysis.png
└── logs/                 # 처리 로그
```

### 결과 예시 (summary.json)

```json
{
  "task_type": "classification",
  "best_model": "xgb",
  "best_synthesis": "smote",
  "performance": 0.8542,
  "data_shape": [150, 4]
}
```

## ⚙️ 설정 옵션

`config.yaml` 파일을 통해 다음을 설정할 수 있습니다:

- 사용할 모델 목록
- 데이터 합성 방법
- 하이퍼파라미터 최적화 설정
- 평가 지표 선택
- 시각화 옵션

## 🎯 사용 사례

### 1. 소규모 분류 문제
```python
# 고객 이탈 예측, 이메일 스팸 분류 등
python main.py --single_file customer_churn.csv --output_folder ./results
```

### 2. 회귀 문제
```python
# 집값 예측, 매출 예측 등
python main.py --single_file house_prices.csv --output_folder ./results
```

### 3. 배치 분석
```python
# 여러 데이터셋 한번에 분석
python main.py --input_folder ./multiple_datasets --output_folder ./batch_results
```

## 📈 성능 최적화

### 소량 데이터를 위한 최적화
- 데이터 증강을 통한 성능 향상
- 적절한 교차 검증 전략
- 과적합 방지 기법 적용

### 처리 속도 향상
- 병렬 처리 지원
- 메모리 효율적 처리
- 단계별 캐싱

## 🔍 문제 해결

### 일반적인 오류들

1. **패키지 누락 오류**
   ```bash
   pip install -r requirements.txt
   ```

2. **메모리 부족**
   - 설정에서 배치 크기 조정
   - 큰 파일의 경우 청크 단위 처리

3. **타겟 컬럼 인식 실패**
   - 컬럼명을 'target', 'label', 'y' 등으로 변경
   - 마지막 컬럼이 타겟으로 자동 인식됨

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License

## 📞 지원

문제가 있거나 질문이 있으시면 이슈를 등록해주세요.

---

**AutoML System** - 소량 데이터를 위한 전문적인 머신러닝 자동화 솔루션
