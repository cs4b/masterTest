from drugex.training.scorers.interfaces import Scorer
from rdkit import Chem
import time

class ModelScorer(Scorer):
    def __init__(self, scoring_function, time_budget=None, sample_budget=None):
        super().__init__()
        self.dh_scorer      = scoring_function
        self.time_budget    = time_budget
        self.sample_budget  = sample_budget
        self.eval_count     = 0
        self._start_time    = None

        # Exposed for monitoring/HP tuning
        self.elapsed_time   = 0.0

    def getKey(self):
        return "DiverseHitsScorer"

    def reset_budgets(self):
        """Call this once before your first training-batch scoring."""
        self.eval_count   = 0
        self._start_time  = None
        self.elapsed_time = 0.0

    def getScores(self, mols, frags=None):
        # 1) Start timer on first call
        if self._start_time is None:
            self._start_time = time.monotonic()

        # 2) Check budgets
        now = time.monotonic()
        self.elapsed_time = now - self._start_time
        if self.time_budget is not None and self.elapsed_time > self.time_budget:
            raise TimeoutError(
                f"Time budget of {self.time_budget}s exceeded "
                f"(elapsed {self.elapsed_time:.2f}s)"
            )
        if self.sample_budget is not None and self.eval_count >= self.sample_budget:
            raise RuntimeError(
                f"Sample budget of {self.sample_budget} reached "
                f"({self.eval_count} molecules scored)"
            )

        # 3) Convert mol objects to SMILES
        smiles_list = []
        for m in mols:
            if m is not None:
                try:
                    smiles_list.append(Chem.MolToSmiles(m))
                except:
                    smiles_list.append(None)
            else:
                smiles_list.append(None)

        # 4) Delegate to the real diverse-hits scorer
        scores = self.dh_scorer(smiles_list)

        # 5) Update sample counter
        #    assume one score per molecule in `mols`
        self.eval_count += len(smiles_list)

        return scores
"""
class ModelScorer(Scorer):
    def __init__(self, scoring_function):
        super().__init__()
        self.dh_scorer = scoring_function

    def getScores(self,mols,frags=None):
        smiles_list = []
        for m in mols:
            if m is not None:
                try:
                    smi = Chem.MolToSmiles(m)
                    smiles_list.append(smi)
                except:
                    smiles_list.append(None)
            else:
                smiles_list.append(None)
        scores = self.dh_scorer(smiles_list)        
        return scores
    
    def getKey(self):
        return "DiverseHitsScorer"
"""