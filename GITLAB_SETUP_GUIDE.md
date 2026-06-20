# GitLab Upload Preparation Guide

This folder contains everything you need to prepare your repository for GitLab upload.

## Files in This Directory

1. **GITLAB_CLEANUP_ANALYSIS.md** - Detailed analysis of your repository structure, which folders are duplicates, and cleanup recommendations
2. **environment.yml** - Conda environment file for GPU (CUDA 11.8) systems
3. **environment-cpu.yml** - Conda environment file for CPU-only systems
4. **RECOMMENDED-.gitignore** - Enhanced .gitignore to prevent large files from being committed

## Quick Start (Recommended - Option B)

This keeps your source models and SMILES data but removes generated results.

### Step 1: Backup Your Work
```bash
cd /system/user/studentwork/nemeth
tar -czf nemeth_backup_$(date +%Y%m%d).tar.gz thesis/
```

### Step 2: Remove Top-Level Duplicates (Frees 8.3 GB)
```bash
rm -rf diverse-hits/
rm -rf DrugEx/
```

### Step 3: Clean Python Cache in thesis/ (Frees ~50 MB)
```bash
cd thesis/
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
```

### Step 4: Remove Generated Results (Optional - Frees ~500 MB)
```bash
# Only if you don't need the results directories for reference
rm -rf thesis/results/
rm -rf thesis/backresults/
```

### Step 5: Update .gitignore
```bash
cd thesis/
# Backup existing
cp .gitignore .gitignore.backup

# Copy the recommended version from the parent directory
cp ../.gitignore .gitignore
# OR manually add the entries from RECOMMENDED-.gitignore
```

### Step 6: Create/Test Conda Environment
```bash
# With CUDA GPU (recommended for your system):
conda env create -f environment.yml

# OR CPU-only:
conda env create -f environment-cpu.yml

# Activate and test
conda activate nemeth-thesis
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from rdkit import Chem; print('RDKit: OK')"
cd thesis/scripts && python -c "from divopt.scoring import BenchmarkScoringFunction; print('divopt: OK')"
```

### Step 7: Verify Before Upload
```bash
cd /system/user/studentwork/nemeth

# Check final repository size
du -sh thesis/

# Count any files larger than 50MB (should be minimal)
find thesis/ -type f -size +50M | sort -k5 -h

# Verify git status
cd thesis/
git status
```

### Step 8: Push to GitLab
```bash
cd thesis/

# Add environment file
git add environment.yml environment-cpu.yml ../environment.yml

# Commit cleanup changes
git commit -m "Prepare for GitLab: remove duplicates, add environment files, update gitignore"

# Push to GitLab
git push origin main  # or your default branch
```

## Additional Cleanup Options

### Remove All Large Files (Most Aggressive - Option A)
```bash
cd thesis/

# Remove all model files
find diverse-hits/optimizers/ -name "*.pkg" -delete

# Remove SMILES data (can be re-downloaded)
rm -f diverse-hits/data/guacamol_v1*.smiles

# Remove results
rm -rf results/ backresults/

# Find any remaining large files and decide
find . -size +50M -type f | sort -k5 -h
```

### Keep Everything But Archive (Option C)
Create a separate "data" branch or archive:
```bash
# Create a branch for large files
git checkout --orphan data-archive
git add -A
git commit -m "Archive of large data files and results"
git push origin data-archive

# Switch back to main dev branch
git checkout main
# Remove large files as needed
rm -rf results/ large_data_files/
git add -A
git commit -m "Remove large files from main branch"
```

## Folder-by-Folder Cleanup Guide

### Keep ✓
- `thesis/scripts/` - Your working Python scripts (ESSENTIAL)
- `thesis/diverse-hits/divopt/` - Source code (ESSENTIAL)  
- `thesis/Drugex/drugex/` - Source code (ESSENTIAL)
- `thesis/diverse-hits/data/scoring_functions/` - Scoring models (RECOMMENDED)
- `thesis/diverse-hits/data/global_settings.json` - Config (SMALL)
- `thesis/.git/` - Version history (git handles this)

### Consider Removing ✗
- `thesis/results/` - Generated optimization results (~300 MB) - OLD RUN DATA
- `thesis/backresults/` - Backup results (~500 MB) - OLD RUN DATA
- `thesis/diverse-hits/data/guacamol_v1*.smiles` (1.2 GB) - Can be re-downloaded
- `thesis/diverse-hits/optimizers/drugex/monitor/*.pkg` (2.5 GB) - Old model checkpoints
- `thesis/Drugex/tutorial/` - Training tutorials (NOT NEEDED)

### Must Remove ✗
- Top-level `/diverse-hits/` directory (DUPLICATE - 6.9 GB)
- Top-level `/DrugEx/` directory (DUPLICATE - 1.4 GB)

## Repository Size After Cleanup

| Scenario | Final Size | Free Space |
|----------|-----------|-----------|
| Minimal (Option A) | ~1.0 GB | ~8.5 GB |
| Recommended (Option B) | ~3.0 GB | ~6.5 GB |
| Keep Everything (Option C) | ~8.0 GB | ~1.5 GB |

## Environment Files Explained

### environment.yml (GPU)
- Python 3.11.3 with PyTorch 2.0.1 (CUDA 11.8)
- All dependencies: PyDGL, RDKit, scikit-learn, scipy, pandas, etc.
- Recommended for your system (has NVIDIA GPU)

### environment-cpu.yml
- Same as above but PyTorch CPU-only
- Use only if GPU is not available

## Verification Checklist

Before pushing to GitLab, verify:

- [ ] Top-level `diverse-hits/` and `DrugEx/` removed
- [ ] `thesis/` is the working repository
- [ ] Python cache cleaned (`__pycache__/` removed)
- [ ] Large optional files decided (remove/keep results, models, etc.)
- [ ] `.gitignore` updated to prevent large files
- [ ] `environment.yml` at root and tested
- [ ] All imports still work in thesis/scripts/
- [ ] `git status` shows only wanted changes
- [ ] Final size checked with `du -sh thesis/`

## Troubleshooting

### Import errors after cleanup?
```bash
cd thesis/scripts/
python seqtest_hp_circles.py  # or test script
# If imports fail, add to sys.path in scripts or set PYTHONPATH
export PYTHONPATH=/system/user/studentwork/nemeth/thesis:$PYTHONPATH
```

### Environment won't create?
```bash
# Check conda channels
conda config --show channels

# If DGL channel issues, install DGL separately
conda install -c dglteam/label/cu117 dgl

# Or use pip
pip install dgl torch
```

### Large files still appearing?
```bash
# Check what's actually being tracked
cd thesis/
git ls-files | grep -E '\.(pkg|smiles|tsv)$'

# Remove from git history (advanced)
git filter-branch --tree-filter 'rm -f path/to/large/file' HEAD
```

## Questions?

Refer back to `GITLAB_CLEANUP_ANALYSIS.md` for detailed technical explanations of the repository structure and recommendations.

---

**Last Updated:** 2024
**Repository:** /system/user/studentwork/nemeth
**Main Working Directory:** thesis/
