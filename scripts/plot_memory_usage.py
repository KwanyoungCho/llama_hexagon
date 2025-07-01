#!/usr/bin/env python3
"""
Memory Usage Visualization Script for QNN Inference
Analyzes memory log files and creates comprehensive graphs

Usage: python3 plot_memory_usage.py <log_file> [output_dir]
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot memory usage from QNN inference logs')
    parser.add_argument('log_file', help='Path to the CSV log file')
    parser.add_argument('--output-dir', '-o', default='./memory_plots', 
                       help='Output directory for plots (default: ./memory_plots)')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output images')
    parser.add_argument('--style', choices=['default', 'seaborn', 'ggplot'], default='seaborn',
                       help='Plot style (default: seaborn)')
    return parser.parse_args()

def load_memory_data(log_file):
    """Load and preprocess memory data from CSV file"""
    try:
        df = pd.read_csv(log_file)
        
        # Check if timestamp column exists and has valid data
        if 'timestamp' not in df.columns:
            print("Error: 'timestamp' column not found in CSV file")
            sys.exit(1)
        
        # Handle potential timestamp issues
        print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Use index-based relative time if timestamps are problematic
        if df['timestamp'].min() < 0 or (df['timestamp'].max() - df['timestamp'].min()) > 1e10:
            print("Warning: Timestamps appear to be invalid, using index-based timing")
            df['relative_time'] = df.index * 0.1  # Assuming 0.1s sampling interval
        else:
            # Convert timestamp to datetime and calculate relative time
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['relative_time'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()
            except:
                print("Warning: Failed to parse timestamps, using index-based timing")
                df['relative_time'] = df.index * 0.1  # Fallback to index-based timing
        
        # Ensure relative_time is not negative and monotonic
        if df['relative_time'].min() < 0:
            df['relative_time'] = df['relative_time'] - df['relative_time'].min()
        
        # Convert KB to MB for better readability
        memory_columns = ['total_memory_kb', 'free_memory_kb', 'available_memory_kb', 
                         'buffers_kb', 'cached_kb', 'swap_total_kb', 'swap_free_kb',
                         'proc_rss_kb', 'proc_vss_kb']
        
        for col in memory_columns:
            if col in df.columns:
                df[col.replace('_kb', '_mb')] = df[col] / 1024
        
        # Calculate used memory
        df['used_memory_mb'] = df['total_memory_mb'] - df['free_memory_mb']
        df['swap_used_mb'] = df['swap_total_mb'] - df['swap_free_mb']
        
        # Ensure we have valid numpy arrays for plotting
        df['relative_time'] = df['relative_time'].values
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def create_system_memory_plot(df, output_dir, format_type, dpi):
    """Create system memory usage plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to numpy arrays to avoid pandas indexing issues
    time_data = df['relative_time'].values
    
    # Plot 1: Overall system memory
    ax1.plot(time_data, df['total_memory_mb'].values, label='Total', linewidth=2, alpha=0.8)
    ax1.plot(time_data, df['used_memory_mb'].values, label='Used', linewidth=2)
    ax1.plot(time_data, df['free_memory_mb'].values, label='Free', linewidth=2)
    ax1.plot(time_data, df['available_memory_mb'].values, label='Available', linewidth=2)
    
    ax1.set_title('System Memory Usage During QNN Inference', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory (MB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory breakdown
    ax2.plot(time_data, df['buffers_mb'].values, label='Buffers', linewidth=2)
    ax2.plot(time_data, df['cached_mb'].values, label='Cached', linewidth=2)
    if df['swap_total_mb'].max() > 0:
        ax2.plot(time_data, df['swap_used_mb'].values, label='Swap Used', linewidth=2)
    
    ax2.set_title('Memory Details', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Memory (MB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / f'system_memory.{format_type}'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"System memory plot saved: {output_file}")
    plt.close()

def create_process_memory_plot(df, output_dir, format_type, dpi):
    """Create process-specific memory usage plot"""
    # Filter data where process was running
    process_df = df[df['proc_rss_mb'] > 0].copy()
    
    if len(process_df) == 0:
        print("Warning: No process memory data found")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Convert to numpy arrays
    proc_time_data = process_df['relative_time'].values
    
    # Plot 1: Process memory usage
    ax1.plot(proc_time_data, process_df['proc_rss_mb'].values, 
             label='RSS (Physical)', linewidth=2, color='red')
    ax1.plot(proc_time_data, process_df['proc_vss_mb'].values, 
             label='VSS (Virtual)', linewidth=2, color='blue', alpha=0.7)
    
    ax1.set_title('Process Memory Usage (llama-cli)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory (MB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory efficiency
    if len(process_df) > 1:
        efficiency = process_df['proc_rss_mb'].values / process_df['proc_vss_mb'].values * 100
        ax2.plot(proc_time_data, efficiency,
                label='Memory Efficiency (RSS/VSS %)', linewidth=2, color='green')
        
        ax2.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Efficiency (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / f'process_memory.{format_type}'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Process memory plot saved: {output_file}")
    plt.close()

def create_memory_pressure_plot(df, output_dir, format_type, dpi):
    """Create memory pressure analysis plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to numpy arrays
    time_data = df['relative_time'].values
    
    # Calculate memory pressure metrics
    total_memory = df['total_memory_mb'].iloc[0]
    memory_pressure = (df['used_memory_mb'].values / total_memory) * 100
    
    # Color zones
    ax.fill_between(time_data, 0, memory_pressure, 
                   where=(memory_pressure < 70), alpha=0.3, color='green', label='Low Pressure (<70%)')
    ax.fill_between(time_data, 0, memory_pressure, 
                   where=(memory_pressure >= 70) & (memory_pressure < 85), 
                   alpha=0.3, color='yellow', label='Medium Pressure (70-85%)')
    ax.fill_between(time_data, 0, memory_pressure, 
                   where=(memory_pressure >= 85), alpha=0.3, color='red', label='High Pressure (>85%)')
    
    ax.plot(time_data, memory_pressure, linewidth=2, color='black', alpha=0.8)
    
    ax.set_title('Memory Pressure During QNN Inference', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Memory Pressure (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Peak Pressure: {memory_pressure.max():.1f}%\n"
    stats_text += f"Avg Pressure: {memory_pressure.mean():.1f}%\n"
    stats_text += f"Total Memory: {total_memory:.0f} MB"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = Path(output_dir) / f'memory_pressure.{format_type}'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Memory pressure plot saved: {output_file}")
    plt.close()

def create_combined_overview_plot(df, output_dir, format_type, dpi):
    """Create combined overview plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to numpy arrays
    time_data = df['relative_time'].values
    
    # Plot 1: System memory timeline
    ax1.plot(time_data, df['used_memory_mb'].values, label='Used', linewidth=2)
    ax1.plot(time_data, df['free_memory_mb'].values, label='Free', linewidth=2)
    ax1.set_title('System Memory', fontweight='bold')
    ax1.set_ylabel('Memory (MB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Process memory
    process_df = df[df['proc_rss_mb'] > 0]
    if len(process_df) > 0:
        proc_time_data = process_df['relative_time'].values
        ax2.plot(proc_time_data, process_df['proc_rss_mb'].values, 
                label='RSS', linewidth=2, color='red')
        ax2.plot(proc_time_data, process_df['proc_vss_mb'].values, 
                label='VSS', linewidth=2, color='blue', alpha=0.7)
    ax2.set_title('Process Memory', fontweight='bold')
    ax2.set_ylabel('Memory (MB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory distribution (pie chart at peak usage)
    peak_idx = df['used_memory_mb'].idxmax()
    peak_data = df.loc[peak_idx]
    
    sizes = [peak_data['used_memory_mb'], peak_data['free_memory_mb']]
    labels = ['Used', 'Free']
    colors = ['#ff9999', '#66b3ff']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Memory Distribution at Peak\n({peak_data["used_memory_mb"]:.0f}MB used)', 
                 fontweight='bold')
    
    # Plot 4: Memory statistics
    stats = {
        'Peak Used': f"{df['used_memory_mb'].max():.0f} MB",
        'Min Free': f"{df['free_memory_mb'].min():.0f} MB",
        'Total': f"{df['total_memory_mb'].iloc[0]:.0f} MB",
        'Duration': f"{df['relative_time'].values[-1]:.1f} sec"
    }
    
    if len(process_df) > 0:
        stats['Peak RSS'] = f"{process_df['proc_rss_mb'].max():.0f} MB"
        stats['Peak VSS'] = f"{process_df['proc_vss_mb'].max():.0f} MB"
    
    ax4.axis('off')
    table_data = [[k, v] for k, v in stats.items()]
    table = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax4.set_title('Memory Statistics', fontweight='bold', pad=20)
    
    # Common formatting
    for ax in [ax1, ax2]:
        ax.set_xlabel('Time (seconds)')
    
    plt.suptitle('QNN Inference Memory Analysis Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path(output_dir) / f'memory_overview.{format_type}'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Overview plot saved: {output_file}")
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n=== Memory Usage Analysis ===")
    
    # System memory stats
    print(f"Total System Memory: {df['total_memory_mb'].iloc[0]:.0f} MB")
    print(f"Peak Used Memory: {df['used_memory_mb'].max():.0f} MB ({df['used_memory_mb'].max()/df['total_memory_mb'].iloc[0]*100:.1f}%)")
    print(f"Minimum Free Memory: {df['free_memory_mb'].min():.0f} MB")
    print(f"Average Free Memory: {df['free_memory_mb'].mean():.0f} MB")
    
    # Process memory stats
    process_df = df[df['proc_rss_mb'] > 0]
    if len(process_df) > 0:
        print(f"\nProcess Memory:")
        print(f"Peak RSS: {process_df['proc_rss_mb'].max():.0f} MB")
        print(f"Peak VSS: {process_df['proc_vss_mb'].max():.0f} MB")
        print(f"Average RSS: {process_df['proc_rss_mb'].mean():.0f} MB")
        print(f"Memory Efficiency (RSS/VSS): {process_df['proc_rss_mb'].mean()/process_df['proc_vss_mb'].mean()*100:.1f}%")
    
    # Timing stats
    duration = df['relative_time'].values[-1]
    print(f"\nTiming:")
    print(f"Total Duration: {duration:.1f} seconds")
    if duration > 0:
        print(f"Sampling Rate: {len(df)/duration:.1f} samples/second")
    else:
        print(f"Sampling Rate: Unable to calculate")

def main():
    args = parse_arguments()
    
    # Check if log file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found")
        sys.exit(1)
    
    # Set matplotlib style
    if args.style != 'default':
        try:
            plt.style.use(args.style)
        except OSError:
            print(f"Warning: Style '{args.style}' not available, using default")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading memory data from: {args.log_file}")
    df = load_memory_data(args.log_file)
    
    print(f"Loaded {len(df)} data points")
    print_summary_statistics(df)
    
    print(f"\nGenerating plots in: {output_dir}")
    
    # Create all plots
    create_system_memory_plot(df, output_dir, args.format, args.dpi)
    create_process_memory_plot(df, output_dir, args.format, args.dpi)
    create_memory_pressure_plot(df, output_dir, args.format, args.dpi)
    create_combined_overview_plot(df, output_dir, args.format, args.dpi)
    
    print(f"\nAll plots saved in: {output_dir}")
    print("Generated files:")
    for plot_file in output_dir.glob(f"*.{args.format}"):
        print(f"  - {plot_file}")

if __name__ == "__main__":
    main() 