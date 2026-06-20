#!/usr/bin/env python3
"""
Repository setup validation script.
Checks that all necessary files are present and properly configured.
"""

import os
import sys
import json
from pathlib import Path

def check_files_present():
    """Verify all critical files exist"""
    print("=" * 60)
    print("CHECKING REPOSITORY FILES")
    print("=" * 60)
    
    critical_files = [
        ".gitignore",
        "environment-unified.yml",
        "README.md",
        "diverse-hits/README.md",
        "diverse-hits/pyproject.toml",
        "DrugEx/README.md",
        "DrugEx/pyproject.toml",
    ]
    
    all_present = True
    for file in critical_files:
        path = Path(file)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_present = False
    
    return all_present

def check_gitignore_patterns():
    """Verify .gitignore contains important patterns"""
    print("\n" + "=" * 60)
    print("CHECKING .gitignore PATTERNS")
    print("=" * 60)
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        print("  ✗ .gitignore not found!")
        return False
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    required_patterns = [
        "*.pkg",          # DrugEx models
        "*.pt",           # PyTorch models
        "*.pkl",          # Pickle files
        "results/",       # Results directory
        "runs/",          # Training runs
        "__pycache__/",   # Python cache
        ".vscode/",       # IDE
        ".ipynb_checkpoints/",  # Jupyter
    ]
    
    all_present = True
    for pattern in required_patterns:
        present = pattern in content
        status = "✓" if present else "✗"
        print(f"  {status} Pattern: {pattern}")
        if not present:
            all_present = False
    
    return all_present

def check_environment_structure():
    """Verify environment file structure"""
    print("\n" + "=" * 60)
    print("CHECKING ENVIRONMENT FILE STRUCTURE")
    print("=" * 60)
    
    env_path = Path("environment-unified.yml")
    if not env_path.exists():
        print("  ✗ environment-unified.yml not found!")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    required_sections = [
        "name: drug-discovery-bench",
        "channels:",
        "dependencies:",
        "pytorch",
        "pytorch=",
        "rdkit",
        "numpy",
        "scipy",
        "pandas",
        "pip:",
        "deepsmiles",
    ]
    
    all_present = True
    for section in required_sections:
        present = section in content
        status = "✓" if present else "✗"
        print(f"  {status} Section: {section}")
        if not present:
            all_present = False
    
    return all_present

def check_readme_structure():
    """Verify README contains key sections"""
    print("\n" + "=" * 60)
    print("CHECKING README.md STRUCTURE")
    print("=" * 60)
    
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("  ✗ README.md not found!")
        return False
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    required_sections = [
        "## Quick Start",
        "## Repository Structure",
        "## Usage",
        "## Testing",
        "### Prerequisites",
        "environment-unified.yml",
        "GuacaMol",
        "DrugEx",
    ]
    
    all_present = True
    for section in required_sections:
        present = section in content
        status = "✓" if present else "✗"
        print(f"  {status} Section: {section}")
        if not present:
            all_present = False
    
    return all_present

def check_directory_structure():
    """Verify key directories exist"""
    print("\n" + "=" * 60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    key_dirs = [
        "diverse-hits",
        "diverse-hits/scripts",
        "diverse-hits/divopt",
        "diverse-hits/optimizers",
        "diverse-hits/data",
        "diverse-hits/notebooks",
        "DrugEx",
        "DrugEx/drugex",
        "thesis",
        "thesis/scripts",
    ]
    
    all_present = True
    for dir_path in key_dirs:
        path = Path(dir_path)
        exists = path.is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}/")
        if not exists:
            all_present = False
    
    return all_present

def check_large_files():
    """Identify large files that should be in .gitignore"""
    print("\n" + "=" * 60)
    print("CHECKING FOR LARGE FILES (should be in .gitignore)")
    print("=" * 60)
    
    large_extensions = ['.pkg', '.pt', '.pth', '.pkl', '.h5']
    large_dirs = ['results', 'backresults', 'runs']
    
    found_files = []
    
    for ext in large_extensions:
        for file in Path(".").rglob(f"*{ext}"):
            if '.git' not in file.parts:
                size_mb = file.stat().st_size / (1024 * 1024)
                found_files.append((str(file), size_mb))
    
    if found_files:
        print(f"\n  Found {len(found_files)} large files:")
        for file, size in sorted(found_files, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    • {file} ({size:.1f} MB)")
    else:
        print("  ✓ No large model/data files found in current directory")
    
    return True

def main():
    """Run all validation checks"""
    os.chdir(Path(__file__).parent)
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  REPOSITORY SETUP VALIDATION".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        "Files Present": check_files_present(),
        ".gitignore Patterns": check_gitignore_patterns(),
        "Environment Structure": check_environment_structure(),
        "README Structure": check_readme_structure(),
        "Directory Structure": check_directory_structure(),
        "Large Files Check": check_large_files(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check}")
    
    print()
    if all_passed:
        print("✓ ALL CHECKS PASSED! Repository is ready for GitLab upload.")
        print("\nNext steps:")
        print("  1. Review the README.md for accuracy")
        print("  2. Test environment setup: conda env create -f environment-unified.yml")
        print("  3. Commit changes: git add . && git commit -m 'Setup for GitLab'")
        print("  4. Push to GitLab")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
