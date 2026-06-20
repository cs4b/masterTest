# GitLab Upload Preparation - Repository Analysis & Cleanup Plan

## Repository Structure Summary

Your workspace contains **3 separate git repositories**:

```
/system/user/studentwork/nemeth/
├── diverse-hits/           (6.9 GB) ✗ DUPLICATE - REMOVE
├── DrugEx/                 (1.4 GB) ✗ DUPLICATE - REMOVE  
├── thesis/                 (WORKING REPO) ✓ KEEP
│   ├── diverse-hits/       (3.0 GB) - Contains code & data
│   ├── Drugex/             (1.3 GB) - Contains code & dependencies
│   ├── scripts/            (YOUR WORKING SCRIPTS) ✓ KEEP
│   ├── results/            (GENERATED RESULTS) → CONSIDER REMOVING OR ARCHIVING
│   ├── backresults/        (BACKUP RESULTS) → CONSIDER REMOVING
│   └── .git/               (Git history)
└── generated_molecules.smi (Small file)
```

## Analysis: Which Folders Are Actually Used?

### ✓ Code Usage Pattern
Your scripts in `/thesis/scripts/` import from and reference:
- `from divopt.scoring import ...` → resolves to `/thesis/diverse-hits/divopt/`
- `from drugex.training...` → resolves to `/thesis/Drugex/drugex/` (installed or available)
- `BASE_DATASETS_PATH = 'diverse-hits/optimizers/drugex'` → `/thesis/diverse-hits/optimizers/drugex/`
- `scoring_dir: 'diverse-hits/data/scoring_functions/drd2'` → `/thesis/diverse-hits/data/scoring_functions/drd2/`

**Conclusion:** Your code uses `/thesis/` as the working repository with embedded dependencies.

### ✗ Duplicate Folders
- `/diverse-hits/` (top-level) - **REMOVE** (6.9 GB) - This is a duplicate of `/thesis/diverse-hits/`
- `/DrugEx/` (top-level) - **REMOVE** (1.4 GB) - This is a duplicate of `/thesis/Drugex/`

These top-level folders are likely the "upstream" repositories before they were copied into thesis/. They contain extra data/results not needed for your working scripts.

## Large Files to Clean Up

### Files >100 MB in `/thesis/` (the one to keep):

#### Category 1: Model Files (.pkg) - ~2.5 GB total
Location: `/thesis/diverse-hits/optimizers/drugex/monitor/` and related
- drd2_FM.pkg, drd2_FM_time.pkg, drd2_FM_onlytime.pkg
- gsk3_FM.pkg, gsk3_FM_time.pkg, gsk3_FM_onlytime.pkg  
- jnk3_FM.pkg, jnk3_FM_time.pkg, jnk3_FM_timeonly.pkg
- Papyrus05.5_graph_trans_PT.pkg
- Various pretrained models

**Action:** Keep if needed for your results. If these can be downloaded/retrained, remove them.

#### Category 2: SMILES Data Files - ~1.2 GB
Location: `/thesis/diverse-hits/data/`
- guacamol_v1_all.smiles (~350MB)
- guacamol_v1_train.smiles (~400MB)
- guacamol_v1_all_maxmin_order.smiles (~350MB)

**Action:** Keep if essential for your code. Otherwise can be re-downloaded.

#### Category 3: Git LFS Objects in diverse-hits/.git/
- ~10+ files >100MB each in `.git/lfs/objects/`
- Will be removed along with top-level `/diverse-hits/` folder

#### Category 4: TSV Files in `/thesis/backresults/` - ~800 MB
- `drd2_train.tsv`, `gsk3_train.tsv` (large encoded files)

**Action:** Archive or remove these backup results.

#### Category 5: Results in `/thesis/results/` - Check actual size
- Contains optimization run results
**Action:** Consider archiving or removing old results not needed for reproducibility.

## Cleanup Strategy

### PHASE 1: Remove Top-Level Duplicates (SAFE - 8.3GB freed)
```bash
# These are complete duplicates - safe to remove
rm -rf /system/user/studentwork/nemeth/diverse-hits/
rm -rf /system/user/studentwork/nemeth/DrugEx/
# Freed: ~8.3 GB
```

### PHASE 2: Create Conda Environment File (Required)
Extract from `/thesis/diverse-hits/envs/divopt.yml` and potentially merge with DrugEx requirements.

### PHASE 3: Clean Up Large Optional Files in `/thesis/`
Choose based on your needs:

**Option A: Minimal Upload (remove everything)**
```bash
# Remove all model files (~2.5 GB)
find thesis/diverse-hits/optimizers/ -name "*.pkg" -delete
# Remove SMILES data (~1.2 GB) - can be re-downloaded
rm thesis/diverse-hits/data/guacamol_v1*.smiles
# Remove results (~500 MB+)
rm -rf thesis/results/ thesis/backresults/
# Freed: ~4.2 GB+
```

**Option B: Keep Source Data, Remove Results**
```bash
# Keep .pkg and .smiles but remove generated results
rm -rf thesis/results/ thesis/backresults/
# Freed: ~500 MB+ (keep your source models)
```

**Option C: Keep Everything (Medium Upload)**
- Keep all model files and data
- Only remove results
- Upload size: ~5-6 GB

### PHASE 4: Remove Python Cache Files (Fast - ~50 MB)
```bash
find thesis/ -type d -name __pycache__ -exec rm -rf {} +
find thesis/ -type f -name "*.pyc" -delete
find thesis/ -type f -name "*.pyo" -delete
```

### PHASE 5: Update .gitignore (Important!)
Add to `/thesis/.gitignore`:
```
# Large model files
*.pkg

# SMILES data (can be re-downloaded)
*.smiles

# Results and backups
results/
backresults/

# Generated molecules
generated_molecules.smi

# Python cache
__pycache__/
*.pyc
*.pyo

# Large data files
*.tsv
data/guacamol_v1*.smiles
```

## Environment File

Create a new conda environment file at the root of your thesis repo:
- Start with `/thesis/diverse-hits/envs/divopt.yml`
- Add packages from `/thesis/Drugex/` and `/thesis/diverse-hits/` dependencies
- Test locally before uploading

## Recommended Actions (In Order)

1. **Backup** your current setup
   ```bash
   cd /system/user/studentwork/nemeth
   tar -czf nemeth_backup_$(date +%Y%m%d).tar.gz thesis/
   ```

2. **Remove top-level duplicates** (Safe, frees 8.3 GB)
   ```bash
   rm -rf diverse-hits/ DrugEx/
   ```

3. **Create environment file** from divopt.yml

4. **Clean Python cache** in thesis/
   ```bash
   find thesis/ -type d -name __pycache__ -exec rm -rf {} +
   ```

5. **Decide on large files** (models, data, results)
   - Option A: Remove all (most aggressive)
   - Option B: Keep models, remove results (recommended)
   - Option C: Add to .gitignore and keep

6. **Create .gitignore** updates

7. **Add environment file** to git
   ```bash
   cp thesis/diverse-hits/envs/divopt.yml thesis/environment.yml
   git add thesis/environment.yml
   ```

8. **Verify** before uploading
   ```bash
   # From thesis directory
   du -sh .  # Check final size
   find . -size +50M | wc -l  # Check for large files remaining
   ```

## Questions to Answer Before Proceeding

1. **Are the .pkg model files needed?** (2.5 GB)
   - If they can be re-downloaded/retrained: REMOVE
   - If they're your trained models: KEEP

2. **Are the guacamol SMILES files needed?** (1.2 GB)
   - If they're standard datasets: REMOVE (can download)
   - If they're preprocessed: KEEP

3. **Do you need the results/ and backresults/ folders?** (500+ MB)
   - If they're for reference only: REMOVE/ARCHIVE
   - If they're part of your thesis: KEEP (but maybe not on GitLab)

4. **Should different folders be separate repos on GitLab?**
   - Option 1: One repo with everything (easier)
   - Option 2: thesis/ as main, diverse-hits/ and DrugEx/ as separate submodules/dependencies

## Files Generated by This Analysis

- This document: `GITLAB_CLEANUP_ANALYSIS.md`
- Recommended conda environment will be: `environment.yml`

---

**Next Steps:** Review this analysis and let me know which cleanup strategy you prefer (A, B, or C), then I can execute it.
