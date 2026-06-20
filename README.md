# Drug Discovery Optimization Benchmark

A comprehensive benchmarking framework for molecular generation and multi-objective optimization algorithms in drug discovery.

## Overview

This repository contains implementations of multiple molecular optimization approaches (DrugEx, GFlowNet, MARS, MIMOSA, REINVENT, SMILES-RNN, and others) evaluated on common benchmarks (GuacaMol, virtual screening tasks). It includes a modular scoring framework (`divopt`) for implementing custom molecular objectives and a diverse hit selection system.

**Key Components:**
- **DrugEx**: Deep learning-based molecular generation with reinforcement learning
- **GFlowNet**: Generative flow networks for diverse molecule exploration  
- **divopt**: Scoring function framework with memory-efficient diversity tracking
- **Benchmarks**: GuacaMol standard benchmarks + custom drug discovery tasks

## Quick Start

### Prerequisites

- **Conda** (Anaconda or Miniconda) - [Install here](https://docs.conda.io/projects/conda/latest/user-guide/install/index.html)
- **Git** (for cloning models and dependencies)
- **GPU** (NVIDIA CUDA 11.8+) - Optional but recommended for performance
- **Disk Space**: ~50 GB (for models, data, and run artifacts)
- **Python 3.11+**

### 1. Set Up the Conda Environment

**For GPU systems (CUDA 11.8+):**
```bash
conda env create -f environment-unified.yml
conda activate drug-discovery-bench
```

**For CPU-only systems:**
Edit `environment-unified.yml` to use CPU PyTorch, then:
```bash
conda env create -f environment-unified.yml
conda activate drug-discovery-bench
```

### 2. Install the Package

```bash
# From the repository root
pip install -e .
```

### 3. Download Data & Models

```bash
cd diverse-hits/
bash setup.sh
```

This script will:
- Download GuacaMol datasets (if not present)
- Extract molecular SMILES data
- Initialize model directories
- Prepare scoring function configurations

### 4. Verify Installation

```bash
# Run basic tests
cd diverse-hits/
python -m pytest test/ -v

# Run a quick example
cd scripts/
python run_single.py --help
```

## Repository Structure

```
.
├── diverse-hits/              # Main benchmark framework
│   ├── divopt/                # Scoring function & memory management
│   ├── optimizers/            # Molecular generation algorithms
│   │   ├── drugex/            # DrugEx implementation
│   │   ├── gflownet_recursion/# GFlowNet optimizer
│   │   ├── guacamol_baselines/# Baseline methods
│   │   └── ...                # Other optimizers
│   ├── scripts/               # Execution scripts
│   │   ├── run_single.py      # Run single optimization task
│   │   ├── run_directory.py   # Batch execution
│   │   └── ...                # Analysis scripts
│   ├── notebooks/             # Analysis & visualization
│   ├── data/                  # Datasets & configurations
│   └── envs/                  # Environment specifications
│
├── DrugEx/                    # DrugEx library
│   ├── drugex/                # Core module
│   ├── data/                  # Pretrained models
│   └── docs/                  # Documentation
│
├── thesis/                    # Benchmarking scripts & results
│   ├── scripts/               # Analysis & training scripts
│   ├── results/               # Generated optimization results
│   └── backresults/           # Backup result archives
│
└── environment-unified.yml    # Conda environment
```

## Usage

### Running Optimizations

**Single optimization run:**
```bash
cd diverse-hits/scripts/
python run_single.py \
  --optimizer drugex \
  --scoring_function drd2 \
  --n_episodes 100 \
  --output_dir results/my_run/
```

**Batch runs across multiple tasks:**
```bash
cd diverse-hits/scripts/
python run_directory.py \
  --config_dir ../data/configs/ \
  --output_dir results/batch_run/
```

### Key Scripts in `thesis/scripts/`:

| Script | Purpose |
|--------|---------|
| `DE_dist_samp_gt_full.py` | DrugEx training with distributed sampling |
| `train.py` | Generic training entry point |
| `extract_metrics.py` | Extract metrics from optimization results |
| `check_budget_constraints.py` | Verify budget compliance |
| `analyze_reward_distributions.py` | Analyze reward signal distributions |
| `ModelScorer.py` | Score molecules with custom functions |

### Jupyter Notebooks

Analysis notebooks are in `diverse-hits/notebooks/`:
- `optimization_curves.ipynb` - Optimization performance comparison
- `algorithm_speeds.ipynb` - Runtime analysis
- `chemical_space.ipynb` - Chemical space coverage analysis
- `hyperparameter_importance.ipynb` - HP sensitivity analysis
- `train_scoring_functions.ipynb` - Train custom scoring functions

Launch JupyterLab:
```bash
cd diverse-hits/
jupyter lab notebooks/
```

## Configuration

### Scoring Functions

Modify `diverse-hits/data/scoring_functions/` to define custom objectives:

```python
from divopt.scoring import BenchmarkScoringFunction

class CustomScorer(BenchmarkScoringFunction):
    def __call__(self, molecules: List[str]) -> np.ndarray:
        """Score molecules (SMILES strings)"""
        scores = []
        for smiles in molecules:
            # Implement your scoring logic
            score = your_metric(smiles)
            scores.append(score)
        return np.array(scores)
```

### Optimizer Parameters

Edit `diverse-hits/data/search_spaces.yaml` or `search_spaces.json` to modify:
- Learning rates
- Batch sizes
- Episode lengths
- Sampling strategies
- Diversity penalties

### Search Spaces

Define chemical constraints in `diverse-hits/data/global_settings.json`:
- Molecular weight limits
- LogP bounds
- Number of rotatable bonds
- SMILES validity checks

## Installing Additional Optimizers

Some optimizers require additional dependencies:

```bash
# GFlowNet
conda env create -f diverse-hits/envs/gflownet.yml

# Virtual Screening Tools
pip install psutil rdkit-contrib
```

## Troubleshooting

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Memory Issues with Large Datasets

- Reduce batch size in configurations
- Use data sampling: `sampling_strategy: random`
- Reduce diversity memory size: `max_memory: 10000`

### RDKit Import Errors

```bash
# Reinstall RDKit
conda remove rdkit
conda install -c conda-forge rdkit
```

### Model Loading Failures

Ensure `.pkg` model files are in correct locations:
```bash
cd diverse-hits/
ls -la optimizers/drugex/*.pkg
```

Download missing models from [GuacaMol data repository](https://github.com/BenevolentAI/guacamol).

## Environment File Details

**environment-unified.yml** includes:
- Python 3.11
- PyTorch 2.0.1 with CUDA 11.8 support
- RDKit (chemistry library)
- DGL (graph neural networks)
- scikit-learn, pandas, numpy, scipy
- JupyterLab, matplotlib, seaborn
- Additional libraries: deepsmiles, selfies, PyTDC, guacamol

**For CPU-only systems**: Edit `environment-unified.yml` to replace the PyTorch lines with CPU-only versions before creating the environment.

## Testing

Run the test suite:

```bash
cd diverse-hits/
pytest test/ -v --tb=short

# Run specific test
pytest test/test_scoring_function.py -v
```

Expected tests:
- Scoring function evaluation
- SMILES validity checks
- Memory management
- Optimizer interface consistency

## Performance Notes

Typical performance on modern GPU (e.g., RTX 3090):
- **DrugEx**: 1,000 molecules/epoch (~5s/epoch)
- **GFlowNet**: 100-500 molecules/epoch (~10-20s)
- **Virtual Screening**: 1M molecule database scan (~30s)

CPU-only execution is ~10-50x slower depending on the algorithm.

## Citation

If you use this framework in your research, please cite:

```bibtex
@thesis{nemeth2026druggenerationbench,
  author={Nemeth, [Your Name]},
  title={Comparative Analysis of Deep Learning Approaches for Molecular Generation and Optimization},
  year={2026},
  school={[University Name]}
}
```

For specific optimizers used, cite their original papers:
- DrugEx: [cite DrugEx paper]
- GFlowNet: [cite GFlowNet paper]
- GuacaMol: Polykovskiy, D., et al. (2020)

## Contributing

Bug reports and feature requests are welcome. Please open an issue or create a pull request.

## License

See LICENSE files in respective directories for license information.

## Contact

For questions or issues, contact: [your-email@example.com]

---

**Last Updated**: June 2026  
**Python Version**: 3.11.3  
**PyTorch Version**: 2.0.1  
**Primary Dependencies**: RDKit, DGL, scikit-learn
