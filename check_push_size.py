#!/usr/bin/env python3
"""
Script to check the size of files that would be pushed to git.
Shows total size and lists the 5 largest files.
"""

import subprocess
import os
from pathlib import Path
from typing import List, Tuple

def get_files_to_push() -> List[str]:
    """Get list of files that would be pushed (tracked by git, not ignored)."""
    result = subprocess.run(
        ['git', 'ls-files'],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    return [f for f in result.stdout.strip().split('\n') if f]

def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def main():
    print("=" * 70)
    print("GIT PUSH SIZE ANALYSIS")
    print("=" * 70)
    
    # Get all tracked files
    print("\n1. Gathering tracked files...")
    files = get_files_to_push()
    print(f"   Found {len(files)} tracked files")
    
    # Calculate sizes
    print("\n2. Calculating sizes...")
    file_sizes: List[Tuple[str, int]] = []
    total_size = 0
    
    for file_path in files:
        size = get_file_size(file_path)
        total_size += size
        file_sizes.append((file_path, size))
    
    # Sort by size
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print("\n" + "=" * 70)
    print("PUSH SIZE SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Size to Push: {format_size(total_size)}")
    print(f"Total Files: {len(files)}")
    
    # Show top 5 largest files
    print("\n" + "-" * 70)
    print("TOP 5 LARGEST FILES")
    print("-" * 70)
    
    for i, (file_path, size) in enumerate(file_sizes[:5], 1):
        size_str = format_size(size)
        pct = (size / total_size * 100) if total_size > 0 else 0
        print(f"{i}. {file_path:<50} {size_str:>12} ({pct:>5.1f}%)")
    
    # Size breakdown by type
    print("\n" + "-" * 70)
    print("SIZE BREAKDOWN BY FILE TYPE")
    print("-" * 70)
    
    type_sizes = {}
    for file_path, size in file_sizes:
        ext = Path(file_path).suffix or 'no_extension'
        if ext not in type_sizes:
            type_sizes[ext] = {'count': 0, 'total': 0}
        type_sizes[ext]['count'] += 1
        type_sizes[ext]['total'] += size
    
    sorted_types = sorted(
        type_sizes.items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )
    
    for ext, data in sorted_types[:10]:
        size_str = format_size(data['total'])
        pct = (data['total'] / total_size * 100) if total_size > 0 else 0
        print(f"{ext:<15} {data['count']:>4} files  {size_str:>12} ({pct:>5.1f}%)")
    
    if len(sorted_types) > 10:
        print(f"... and {len(sorted_types) - 10} more file types")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if total_size < 100 * 1024 * 1024:  # < 100 MB
        status = "✅ SMALL - Safe to push"
    elif total_size < 500 * 1024 * 1024:  # < 500 MB
        status = "✓ MODERATE - Reasonable push size"
    elif total_size < 1024 * 1024 * 1024:  # < 1 GB
        status = "⚠ LARGE - Consider cleanup"
    else:
        status = "✗ VERY LARGE - Cleanup recommended"
    
    print(f"\nStatus: {status}")
    print(f"Push size: {format_size(total_size)}")
    print(f"Files: {len(files)}")
    
    # Check for untracked files
    print("\n" + "-" * 70)
    print("IGNORED FILES CHECK")
    print("-" * 70)
    
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True,
        text=True
    )
    
    untracked = [line for line in result.stdout.split('\n') 
                 if line.startswith('??') and line.strip()]
    
    if untracked:
        print(f"\n⚠ Found {len(untracked)} untracked files:")
        for line in untracked[:5]:
            print(f"   {line}")
        if len(untracked) > 5:
            print(f"   ... and {len(untracked) - 5} more")
    else:
        print("\n✓ No untracked files (all properly ignored)")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
