#!/usr/bin/env python3
"""
AutoML System 사용 예시
Example usage of AutoML System
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# 예시 데이터 생성 함수들
def create_sample_classification_data():
    """분류용 예시 데이터 생성"""
    
    np.random.seed(42)
    n_samples = 200
    
    # 특성 생성
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.uniform(0, 10, n_samples)
    feature4 = np.random.exponential(2, n_samples)
    
    # 범주형 특성
    categories = ['A', 'B', 'C']
    feature5 = np.random.choice(categories, n_samples)
    
    # 타겟 생성 (분류)
    target = np.where(
        (feature1 + feature2 > 0) & (feature3 > 5), 
        1, 
        0
    )
    
    # 일부 노이즈 추가
    noise_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
    target[noise_indices] = 1 - target[noise_indices]
    
    # 데이터프레임 생성
    data = pd.DataFrame({
        'numerical_feature_1': feature1,
        'numerical_feature_2': feature2,
        'numerical_feature_3': feature3,
        'numerical_feature_4': feature4,
        'categorical_feature': feature5,
        'target': target
    })
    
    return data


def create_sample_regression_data():
    """회귀용 예시 데이터 생성"""
    
    np.random.seed(42)
    n_samples = 150
    
    # 특성 생성
    feature1 = np.random.normal(5, 2, n_samples)
    feature2 = np.random.uniform(0, 100, n_samples)
    feature3 = np.random.exponential(1, n_samples)
    
    # 범주형 특성
    categories = ['Type_X', 'Type_Y', 'Type_Z']
    feature4 = np.random.choice(categories, n_samples)
    category_effect = np.where(feature4 == 'Type_X', 10, 
                              np.where(feature4 == 'Type_Y', 5, 0))
    
    # 타겟 생성 (회귀)
    target = (
        2 * feature1 + 
        0.5 * feature2 + 
        10 * feature3 + 
        category_effect + 
        np.random.normal(0, 5, n_samples)  # 노이즈
    )
    
    # 데이터프레임 생성
    data = pd.DataFrame({
        'size': feature1,
        'age': feature2,
        'distance': feature3,
        'type': feature4,
        'price': target
    })
    
    return data


def create_imbalanced_classification_data():
    """불균형 분류 데이터 생성"""
    
    np.random.seed(42)
    n_samples = 300
    
    # 특성 생성
    features = np.random.normal(0, 1, (n_samples, 5))
    
    # 불균형 타겟 생성 (클래스 0이 대부분)
    target = np.zeros(n_samples)
    minority_indices = np.random.choice(n_samples, size=30, replace=False)  # 10%만 클래스 1
    target[minority_indices] = 1
    
    # 소수 클래스에 대해 특성 조정
    features[minority_indices] += 2
    
    # 데이터프레임 생성
    data = pd.DataFrame(features, columns=[f'feature_{i+1}' for i in range(5)])
    data['label'] = target.astype(int)
    
    return data


def setup_example_data():
    """예시 데이터 폴더 구조 생성"""
    
    # 데이터 폴더 생성
    data_folder = Path('example_data')
    data_folder.mkdir(exist_ok=True)
    
    print("🎯 예시 데이터 생성 중...")
    
    # 1. 분류 데이터
    print("   📊 분류 데이터 생성...")
    classification_data = create_sample_classification_data()
    classification_data.to_csv(data_folder / 'classification_example.csv', index=False)
    
    # 2. 회귀 데이터
    print("   📈 회귀 데이터 생성...")
    regression_data = create_sample_regression_data()
    regression_data.to_csv(data_folder / 'regression_example.csv', index=False)
    
    # 3. 불균형 분류 데이터
    print("   ⚖️ 불균형 분류 데이터 생성...")
    imbalanced_data = create_imbalanced_classification_data()
    imbalanced_data.to_csv(data_folder / 'imbalanced_classification.csv', index=False)
    
    # 4. Excel 파일 생성
    print("   📗 Excel 파일 생성...")
    with pd.ExcelWriter(data_folder / 'mixed_data.xlsx') as writer:
        classification_data.to_excel(writer, sheet_name='classification', index=False)
        regression_data.to_excel(writer, sheet_name='regression', index=False)
    
    print(f"✅ 예시 데이터가 '{data_folder}' 폴더에 생성되었습니다!")
    return data_folder


def run_example():
    """예시 실행"""
    
    print("🚀 AutoML System 예시 실행")
    print("="*50)
    
    # 예시 데이터 생성
    data_folder = setup_example_data()
    
    # 결과 폴더 생성
    results_folder = Path('example_results')
    results_folder.mkdir(exist_ok=True)
    
    print(f"\n📁 생성된 데이터 파일들:")
    for file in data_folder.glob('*'):
        print(f"   - {file.name}")
    
    print(f"\n🎯 실행 명령어 예시:")
    print(f"   # 단일 파일 처리")
    print(f"   python main.py --single_file {data_folder}/classification_example.csv --output_folder {results_folder}/classification")
    print(f"   ")
    print(f"   # 배치 처리")
    print(f"   python main.py --input_folder {data_folder} --output_folder {results_folder}/batch")
    print(f"   ")
    print(f"   # 설정 파일 사용")
    print(f"   python main.py --single_file {data_folder}/regression_example.csv --output_folder {results_folder}/regression --config config.yaml")
    
    print(f"\n💡 참고사항:")
    print(f"   - 첫 실행 시 필요한 패키지들이 자동으로 다운로드됩니다")
    print(f"   - 소량 데이터이므로 처리 시간은 1-5분 정도 소요됩니다")
    print(f"   - 결과는 {results_folder} 폴더에 저장됩니다")
    
    # 데이터 미리보기
    print(f"\n📋 데이터 미리보기:")
    
    # 분류 데이터
    classification_data = pd.read_csv(data_folder / 'classification_example.csv')
    print(f"\n   🔸 분류 데이터 (classification_example.csv):")
    print(f"     - 형태: {classification_data.shape}")
    print(f"     - 타겟: {classification_data['target'].value_counts().to_dict()}")
    print(f"     - 특성: {list(classification_data.columns[:-1])}")
    
    # 회귀 데이터
    regression_data = pd.read_csv(data_folder / 'regression_example.csv')
    print(f"\n   🔸 회귀 데이터 (regression_example.csv):")
    print(f"     - 형태: {regression_data.shape}")
    print(f"     - 타겟 범위: {regression_data['price'].min():.1f} ~ {regression_data['price'].max():.1f}")
    print(f"     - 특성: {list(regression_data.columns[:-1])}")
    
    # 불균형 데이터
    imbalanced_data = pd.read_csv(data_folder / 'imbalanced_classification.csv')
    print(f"\n   🔸 불균형 분류 데이터 (imbalanced_classification.csv):")
    print(f"     - 형태: {imbalanced_data.shape}")
    print(f"     - 클래스 분포: {imbalanced_data['label'].value_counts().to_dict()}")
    
    print(f"\n🎉 예시 데이터 준비가 완료되었습니다!")
    print(f"   이제 위의 명령어로 AutoML 시스템을 실행해보세요.")


if __name__ == "__main__":
    run_example()