#!/usr/bin/env python3
"""
Simple Memory Usage Visualization for QNN Inference
간단한 메모리 사용량 시각화 도구
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simple memory usage plotter for QNN inference')
    parser.add_argument('log_file', help='Path to the CSV log file')
    parser.add_argument('--output-dir', '-o', default='./simple_memory_plots', 
                       help='Output directory for plots (default: ./simple_memory_plots)')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output images (default: 150)')
    return parser.parse_args()

def load_memory_data(log_file):
    """Load and preprocess memory data from CSV file"""
    try:
        df = pd.read_csv(log_file)
        
        # Check basic columns
        required_cols = ['timestamp', 'total_memory_kb', 'free_memory_kb']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found")
                sys.exit(1)
        
        # Convert KB to MB
        df['total_memory_mb'] = df['total_memory_kb'] / 1024
        df['free_memory_mb'] = df['free_memory_kb'] / 1024
        df['used_memory_mb'] = df['total_memory_mb'] - df['free_memory_mb']
        
        # Simple time calculation (use index if timestamp is problematic)
        if len(df) > 1:
            df['time_sec'] = np.arange(len(df)) * 0.1  # 0.1초 간격으로 가정
        else:
            df['time_sec'] = [0]
        
        # Calculate memory usage percentage
        df['memory_usage_percent'] = (df['used_memory_mb'] / df['total_memory_mb']) * 100
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def create_simple_memory_plots(df, output_dir, format_type, dpi):
    """Create simple memory usage plots"""
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    time_data = df['time_sec'].values
    
    # Plot 1: Memory Usage Percentage
    ax1.plot(time_data, df['memory_usage_percent'].values, 
             linewidth=3, color='red', label='Memory Usage %')
    
    ax1.set_title('Memory Usage (%)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Usage (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add percentage text
    max_usage = df['memory_usage_percent'].max()
    avg_usage = df['memory_usage_percent'].mean()
    ax1.text(0.02, 0.95, f'Max: {max_usage:.1f}%\nAvg: {avg_usage:.1f}%', 
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Actual Memory Usage (MB)
    total_memory = df['total_memory_mb'].iloc[0]
    ax2.plot(time_data, df['used_memory_mb'].values, 
             linewidth=3, color='blue', label='Used Memory')
    
    # Add total memory line
    ax2.axhline(y=total_memory, color='gray', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Total Memory ({total_memory:.0f} MB)')
    
    ax2.set_title('Actual Memory Usage (MB)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Time (seconds)', fontsize=14)
    ax2.set_ylabel('Memory (MB)', fontsize=14)
    ax2.set_ylim(0, total_memory * 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Add memory text
    max_memory = df['used_memory_mb'].max()
    min_free = df['free_memory_mb'].min()
    ax2.text(0.02, 0.95, f'Peak Used: {max_memory:.0f} MB\nMin Free: {min_free:.0f} MB', 
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_dir) / f'simple_memory_usage.{format_type}'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Simple memory plot saved: {output_file}")
    plt.close()

def print_simple_stats(df):
    """Print simple statistics"""
    total_memory = df['total_memory_mb'].iloc[0]
    max_used = df['used_memory_mb'].max()
    min_free = df['free_memory_mb'].min()
    max_percent = df['memory_usage_percent'].max()
    avg_percent = df['memory_usage_percent'].mean()
    duration = df['time_sec'].values[-1]
    
    print("\n" + "="*50)
    print("           메모리 사용량 분석 결과")
    print("="*50)
    print(f"전체 메모리:     {total_memory:.0f} MB")
    print(f"최대 사용량:     {max_used:.0f} MB ({max_percent:.1f}%)")
    print(f"평균 사용률:     {avg_percent:.1f}%")
    print(f"최소 여유 메모리: {min_free:.0f} MB")
    print(f"측정 시간:       {duration:.1f} 초")
    print(f"데이터 포인트:   {len(df)} 개")
    print("="*50)

def main():
    args = parse_arguments()
    
    # Check if log file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading memory data from: {args.log_file}")
    df = load_memory_data(args.log_file)
    
    print(f"Loaded {len(df)} data points")
    print_simple_stats(df)
    
    print(f"\nGenerating simple plot in: {output_dir}")
    create_simple_memory_plots(df, output_dir, args.format, args.dpi)
    
    print(f"\n✅ 완료! 그래프가 저장되었습니다: {output_dir}/simple_memory_usage.{args.format}")

if __name__ == "__main__":
    main() 