#!/usr/bin/env python3
"""
Main execution script for AutoML System
소량의 데이터를 위한 자동 ML 시스템

Usage:
    python main.py --input_folder ./data --output_folder ./results
    python main.py --single_file ./data/sample.csv --output_folder ./results
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from auto_ml_system import AutoMLPipeline


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='자동 ML 시스템 - 소량 데이터용 분류/회귀 태스크 자동 인식 및 최적화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 폴더 단위 배치 처리
  python main.py --input_folder ./data --output_folder ./results
  
  # 단일 파일 처리
  python main.py --single_file ./data/sample.csv --output_folder ./results
  
  # 설정 파일 사용
  python main.py --input_folder ./data --output_folder ./results --config config.yaml

지원 파일 형식: CSV, Excel (.xlsx, .xls), JSON, Parquet
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_folder', '-i',
        type=str,
        help='입력 데이터 폴더 경로 (배치 처리)'
    )
    input_group.add_argument(
        '--single_file', '-f',
        type=str,
        help='단일 파일 경로'
    )
    
    # Output folder (required)
    parser.add_argument(
        '--output_folder', '-o',
        type=str,
        required=True,
        help='결과 출력 폴더 경로'
    )
    
    # Optional config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='설정 파일 경로 (YAML 또는 JSON)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate inputs
    if args.input_folder and not os.path.exists(args.input_folder):
        print(f"❌ 오류: 입력 폴더가 존재하지 않습니다: {args.input_folder}")
        sys.exit(1)
    
    if args.single_file and not os.path.exists(args.single_file):
        print(f"❌ 오류: 입력 파일이 존재하지 않습니다: {args.single_file}")
        sys.exit(1)
    
    if args.config and not os.path.exists(args.config):
        print(f"❌ 오류: 설정 파일이 존재하지 않습니다: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize AutoML Pipeline
        print("🚀 AutoML 시스템 초기화 중...")
        pipeline = AutoMLPipeline(config_path=args.config)
        
        if args.input_folder:
            # Batch processing
            print(f"📁 배치 처리 시작: {args.input_folder} → {args.output_folder}")
            results = pipeline.run_batch_processing(args.input_folder, args.output_folder)
            
            # Print summary
            successful = sum(1 for r in results.values() if 'error' not in r)
            total = len(results)
            
            print(f"\n✅ 배치 처리 완료!")
            print(f"   📊 총 파일: {total}")
            print(f"   ✅ 성공: {successful}")
            print(f"   ❌ 실패: {total - successful}")
            print(f"   📂 결과 저장: {args.output_folder}")
            
        else:
            # Single file processing
            print(f"📄 단일 파일 처리 시작: {args.single_file} → {args.output_folder}")
            result = pipeline.run_single_file(args.single_file, args.output_folder)
            
            print(f"\n✅ 파일 처리 완료!")
            print(f"   📊 태스크 유형: {result['task_info']['task_type']}")
            print(f"   🏆 최적 모델: {result['best_model']['model_name']}")
            print(f"   🔬 최적 합성: {result['best_model']['synthesis_method']}")
            print(f"   📈 성능 점수: {result['best_model']['performance']:.4f}")
            print(f"   📂 결과 저장: {args.output_folder}")
        
        print(f"\n🎉 모든 작업이 완료되었습니다!")
        print(f"📁 상세 결과는 {args.output_folder} 폴더를 확인하세요.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        print("💡 해결 방법:")
        print("   1. 입력 데이터 형식 확인 (CSV, Excel, JSON, Parquet 지원)")
        print("   2. 데이터에 target/label 컬럼이 있는지 확인")
        print("   3. 필요한 패키지가 설치되었는지 확인: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()