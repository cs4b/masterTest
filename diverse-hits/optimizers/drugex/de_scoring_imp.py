from drugex.training.scorers.interfaces import Scorer
from divopt.scoring import BenchmarkScoringFunction
from drugex.training.environment import DrugExEnvironment
from drugex.training.generators import GraphTransformer
from drugex.data.corpus.vocabulary import VocGraph
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.data.datasets import GraphFragDataSet
from rdkit import Chem

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

#init drugex
MODEL_PATH = "diverse-hits/optimizers/drugex/Papyrus05.5_graph_trans_PT.pkg"  
VOCAB_PATH = "diverse-hits/optimizers/drugex/papvoc.vocab" 
vocab = VocGraph.fromFile(VOCAB_PATH)
GPUS = [0]
pretrained = GraphTransformer(voc_trg=vocab, use_gpus=GPUS)
pretrained.loadStatesFromFile(MODEL_PATH)

#constraints
time_budget = 300
sample_budget = 10000
memory_distance_threshold = 0.7
memory_score_threshold = 0.5

"""
#define scoring function
scoring_function_dir = "diverse-hits/data/scoring_functions/drd2"
scoring_function = BenchmarkScoringFunction(
    scoring_function_dir=scoring_function_dir,
    time_budget=time_budget,
    sample_budget=sample_budget,
    memory_distance_threshold=None,
    memory_score_threshold=None,
    memory_known_active_init=False,
    use_property_constraints=True,
    n_jobs=8,
    print_progress=True
)
scoring_function.start_timer_and_reset()

model_scorer = ModelScorer(scoring_function)
env = DrugExEnvironment(
    scorers=[model_scorer],

)
test_from_file = GraphFragDataSet("diverse-hits/optimizers/drugex/drd2_frags.tsv")
generated = pretrained.generate(input_dataset=test_from_file, evaluator=env, num_samples=1000)

print(generated.columns)
print(generated.head())
"""
#make fragments from data/scoring_functions/ all three.

def scoringFunctionCall(sf_path: str):
    scoring_function = BenchmarkScoringFunction(
    scoring_function_dir=sf_path,
    time_budget=time_budget,
    sample_budget=sample_budget,
    memory_distance_threshold=None,
    memory_score_threshold=None,
    memory_known_active_init=False,
    use_property_constraints=True,
    n_jobs=8,
    print_progress=True
    )
    scoring_function.start_timer_and_reset()

    return scoring_function

model_scorer_drd= ModelScorer(scoringFunctionCall("diverse-hits/data/scoring_functions/drd2"))
env_drd = DrugExEnvironment(scorers=[model_scorer_drd])
bias_drd = GraphFragDataSet("diverse-hits/optimizers/drugex/fragments/drd2_frags.tsv")
generated_drd = pretrained.generate(input_dataset=bias_drd, evaluator=env_drd, num_samples=1000)
print(generated_drd.columns)
print(generated_drd.head())

model_scorer_gsk= ModelScorer(scoringFunctionCall("diverse-hits/data/scoring_functions/gsk3"))
env_gsk = DrugExEnvironment(scorers=[model_scorer_gsk])
bias_gsk = GraphFragDataSet("diverse-hits/optimizers/drugex/fragments/gsk3_frags.tsv")
generated_gsk = pretrained.generate(input_dataset=bias_gsk, evaluator=env_gsk, num_samples=1000)
print(generated_gsk.columns)
print(generated_gsk.head())

model_scorer_jnk= ModelScorer(scoringFunctionCall("diverse-hits/data/scoring_functions/jnk3"))
env_jnk = DrugExEnvironment(scorers=[model_scorer_jnk])
bias_jnk = GraphFragDataSet("diverse-hits/optimizers/drugex/fragments/jnk3_frags.tsv")
generated_jnk = pretrained.generate(input_dataset=bias_jnk, evaluator=env_jnk, num_samples=1000)
print(generated_jnk.columns)
print(generated_jnk.head())

