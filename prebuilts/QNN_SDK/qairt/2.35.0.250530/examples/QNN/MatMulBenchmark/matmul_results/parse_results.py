#!/usr/bin/env python3
"""
MatMul Benchmark Results Parser
각 패턴별로 GFLOPS와 Time(ms) CSV 표를 생성합니다.
Row: Datatype, Column: Sequence Length
"""

import os
import re
import pandas as pd
from typing import Dict, List, Tuple

def parse_result_file(filepath: str) -> Tuple[str, Dict[str, Dict[int, Tuple[float, float]]]]:
    """
    결과 파일을 파싱해서 패턴별 데이터를 추출
    
    Returns:
        datatype: 데이터타입 (fp16, fp32, int8, int16)
        patterns: {pattern_name: {seq_len: (time_ms, gflops)}}
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Datatype 추출
    dtype_match = re.search(r'Input A dtype: (\w+)', content)
    datatype = dtype_match.group(1) if dtype_match else "unknown"
    
    patterns = {}
    
    # 패턴별로 데이터 추출
    pattern_sections = re.split(r'=== (.+?) ===', content)[1:]  # 첫 번째 빈 요소 제거
    
    for i in range(0, len(pattern_sections), 2):
        if i + 1 >= len(pattern_sections):
            break
            
        pattern_name = pattern_sections[i].strip()
        pattern_data = pattern_sections[i + 1]
        
        # 패턴 이름 정규화
        if "Attention" in pattern_name:
            pattern_key = "Attention"
        elif "Linear" in pattern_name:
            pattern_key = "Linear"
        elif "FFN_Up" in pattern_name:
            pattern_key = "FFN_Up"
        elif "FFN_Down" in pattern_name:
            pattern_key = "FFN_Down"
        else:
            continue
        
        patterns[pattern_key] = {}
        
        # 각 라인에서 seq, time, gflops 추출
        lines = pattern_data.split('\n')
        for line in lines:
            # seq length, time, gflops가 있는 라인 찾기
            match = re.match(r'^(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+OK', line.strip())
            if match:
                seq_len = int(match.group(1))
                time_ms = float(match.group(2))
                gflops = float(match.group(3))
                patterns[pattern_key][seq_len] = (time_ms, gflops)
    
    return datatype, patterns

def create_csv_tables(all_data: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]]):
    """
    모든 데이터를 패턴별 CSV 표로 생성
    """
    # 모든 sequence length 수집
    all_seq_lens = set()
    for datatype_data in all_data.values():
        for pattern_data in datatype_data.values():
            all_seq_lens.update(pattern_data.keys())
    
    seq_lens = sorted(all_seq_lens)
    
    # 모든 패턴 수집
    all_patterns = set()
    for datatype_data in all_data.values():
        all_patterns.update(datatype_data.keys())
    
    print("🎯 MatMul Benchmark Results Summary")
    print("=" * 60)
    print(f"Data Types: {list(all_data.keys())}")
    print(f"Patterns: {sorted(all_patterns)}")
    print(f"Sequence Lengths: {seq_lens}")
    print("=" * 60)
    
    # 패턴별로 CSV 생성
    for pattern in sorted(all_patterns):
        print(f"\n📊 Pattern: {pattern}")
        print("-" * 40)
        
        # GFLOPS 표 생성
        gflops_data = {}
        time_data = {}
        
        for datatype in sorted(all_data.keys()):
            if pattern in all_data[datatype]:
                gflops_row = []
                time_row = []
                
                for seq_len in seq_lens:
                    if seq_len in all_data[datatype][pattern]:
                        time_ms, gflops = all_data[datatype][pattern][seq_len]
                        gflops_row.append(gflops)
                        time_row.append(time_ms)
                    else:
                        gflops_row.append(None)  # 데이터 없음
                        time_row.append(None)
                
                gflops_data[datatype] = gflops_row
                time_data[datatype] = time_row
        
        # DataFrame 생성 (row: sequence length, column: datatype)
        gflops_df = pd.DataFrame(gflops_data, index=[f"seq_{seq}" for seq in seq_lens])
        time_df = pd.DataFrame(time_data, index=[f"seq_{seq}" for seq in seq_lens])
        
        # CSV 파일명 생성
        pattern_name_clean = pattern.replace(" ", "_").replace("[", "").replace("]", "")
        gflops_filename = f"{pattern_name_clean}_GFLOPS.csv"
        time_filename = f"{pattern_name_clean}_Time_ms.csv"
        
        # CSV 저장
        gflops_df.to_csv(gflops_filename)
        time_df.to_csv(time_filename)
        
        print(f"✅ Generated: {gflops_filename}")
        print(f"✅ Generated: {time_filename}")
        
        # 테이블 미리보기 출력
        print(f"\n📈 {pattern} - GFLOPS:")
        print(gflops_df.round(2))
        
        print(f"\n⏱️  {pattern} - Time (ms):")
        print(time_df.round(2))

def main():
    """메인 함수"""
    # 결과 파일들 찾기
    result_files = [
        "fp16_fp16.txt",
        "fp32_fp32.txt", 
        "int8_int8.txt",
        "int16_int16.txt"
    ]
    
    all_data = {}
    
    # 각 파일 파싱
    for filename in result_files:
        if os.path.exists(filename):
            print(f"📁 Parsing {filename}...")
            datatype, patterns = parse_result_file(filename)
            all_data[datatype] = patterns
            print(f"   Found {len(patterns)} patterns with datatype {datatype}")
        else:
            print(f"⚠️  File not found: {filename}")
    
    if not all_data:
        print("❌ No data files found!")
        return
    
    # CSV 표 생성
    create_csv_tables(all_data)
    
    print(f"\n🎉 Successfully generated CSV tables for all patterns!")
    print(f"   Total datatypes processed: {len(all_data)}")

if __name__ == "__main__":
    main() 