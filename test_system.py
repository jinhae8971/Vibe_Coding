#!/usr/bin/env python3
"""
AutoML System 테스트 스크립트
Test script for AutoML System
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """필수 패키지 import 테스트"""
    print("🔍 패키지 import 테스트...")
    
    try:
        import pandas as pd
        print("   ✅ pandas")
    except ImportError:
        print("   ❌ pandas - pip install pandas")
        return False
    
    try:
        import numpy as np
        print("   ✅ numpy")
    except ImportError:
        print("   ❌ numpy - pip install numpy")
        return False
    
    try:
        import sklearn
        print("   ✅ scikit-learn")
    except ImportError:
        print("   ❌ scikit-learn - pip install scikit-learn")
        return False
    
    try:
        import yaml
        print("   ✅ pyyaml")
    except ImportError:
        print("   ❌ pyyaml - pip install pyyaml")
        return False
    
    # Optional packages
    optional_packages = [
        ('imblearn', 'imbalanced-learn'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('catboost', 'catboost'),
        ('optuna', 'optuna'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly')
    ]
    
    available_optional = []
    for package_name, install_name in optional_packages:
        try:
            __import__(package_name)
            print(f"   ✅ {package_name}")
            available_optional.append(package_name)
        except ImportError:
            print(f"   ⚠️ {package_name} (선택사항) - pip install {install_name}")
    
    print(f"\n📊 사용 가능한 고급 기능: {len(available_optional)}/{len(optional_packages)}")
    
    return True


def test_automl_import():
    """AutoML 시스템 import 테스트"""
    print("\n🔍 AutoML 시스템 import 테스트...")
    
    try:
        from auto_ml_system import AutoMLPipeline
        print("   ✅ AutoMLPipeline")
        
        from auto_ml_system.data_processor import DataProcessor
        print("   ✅ DataProcessor")
        
        from auto_ml_system.task_detector import TaskDetector
        print("   ✅ TaskDetector")
        
        from auto_ml_system.data_synthesizer import DataSynthesizer
        print("   ✅ DataSynthesizer")
        
        from auto_ml_system.model_trainer import ModelTrainer
        print("   ✅ ModelTrainer")
        
        from auto_ml_system.evaluator import ModelEvaluator
        print("   ✅ ModelEvaluator")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ AutoML 시스템 import 실패: {e}")
        return False


def create_simple_test_data():
    """간단한 테스트 데이터 생성"""
    print("\n🔍 테스트 데이터 생성...")
    
    # 작은 분류 데이터
    np.random.seed(42)
    n_samples = 50
    
    X = np.random.randn(n_samples, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    data = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    data['target'] = y
    
    # 테스트 파일 저장
    test_file = Path('test_data.csv')
    data.to_csv(test_file, index=False)
    
    print(f"   ✅ 테스트 데이터 생성: {test_file}")
    print(f"      - 형태: {data.shape}")
    print(f"      - 타겟 분포: {data['target'].value_counts().to_dict()}")
    
    return test_file


def test_pipeline_initialization():
    """파이프라인 초기화 테스트"""
    print("\n🔍 파이프라인 초기화 테스트...")
    
    try:
        from auto_ml_system import AutoMLPipeline
        
        # 기본 설정으로 초기화
        pipeline = AutoMLPipeline()
        print("   ✅ 기본 설정으로 초기화 성공")
        
        # 설정 파일로 초기화 (존재하는 경우)
        config_file = Path('config.yaml')
        if config_file.exists():
            pipeline_with_config = AutoMLPipeline(str(config_file))
            print("   ✅ 설정 파일로 초기화 성공")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 파이프라인 초기화 실패: {e}")
        return False


def test_data_processing():
    """데이터 처리 테스트"""
    print("\n🔍 데이터 처리 테스트...")
    
    try:
        from auto_ml_system.data_processor import DataProcessor
        
        # 테스트 데이터 생성
        test_file = create_simple_test_data()
        
        # 데이터 프로세서 초기화
        config = {'random_state': 42}
        processor = DataProcessor(config)
        
        # 데이터 로딩
        data = processor.load_data(str(test_file))
        print(f"   ✅ 데이터 로딩 성공: {data.shape}")
        
        # 데이터 전처리
        processed_data = processor.preprocess(data)
        print(f"   ✅ 데이터 전처리 성공")
        print(f"      - X 형태: {processed_data['X'].shape}")
        print(f"      - y 형태: {processed_data['y'].shape}")
        print(f"      - 태스크 타입: {processed_data['target_type']}")
        
        # 정리
        test_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"   ❌ 데이터 처리 실패: {e}")
        return False


def test_task_detection():
    """태스크 감지 테스트"""
    print("\n🔍 태스크 감지 테스트...")
    
    try:
        from auto_ml_system.task_detector import TaskDetector
        from auto_ml_system.data_processor import DataProcessor
        
        # 테스트 데이터 준비
        test_file = create_simple_test_data()
        
        config = {'random_state': 42}
        processor = DataProcessor(config)
        detector = TaskDetector(config)
        
        # 데이터 처리 및 태스크 감지
        data = processor.load_data(str(test_file))
        processed_data = processor.preprocess(data)
        task_info = detector.detect_task(processed_data)
        
        print(f"   ✅ 태스크 감지 성공")
        print(f"      - 태스크 타입: {task_info['task_type']}")
        print(f"      - 샘플 수: {task_info['n_samples']}")
        print(f"      - 특성 수: {task_info['n_features']}")
        print(f"      - 데이터셋 크기: {task_info['dataset_size']}")
        
        # 정리
        test_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"   ❌ 태스크 감지 실패: {e}")
        return False


def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🧪 AutoML System 종합 테스트")
    print("="*50)
    
    tests = [
        ("기본 패키지 import", test_imports),
        ("AutoML 시스템 import", test_automl_import),
        ("파이프라인 초기화", test_pipeline_initialization),
        ("데이터 처리", test_data_processing),
        ("태스크 감지", test_task_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} 테스트 중 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n📊 테스트 결과 요약")
    print("-"*30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 전체 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 통과했습니다! 시스템이 정상적으로 작동할 준비가 되었습니다.")
        print("\n💡 다음 단계:")
        print("   1. python example_usage.py 로 예시 데이터를 생성하세요")
        print("   2. python main.py --help 로 사용법을 확인하세요")
        print("   3. 실제 데이터로 시스템을 테스트해보세요")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        print("💡 해결 방법:")
        print("   1. pip install -r requirements.txt 로 모든 의존성을 설치하세요")
        print("   2. Python 3.7 이상 버전을 사용하고 있는지 확인하세요")
        print("   3. 오류 메시지를 확인하여 문제를 해결하세요")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)