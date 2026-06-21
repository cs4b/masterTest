import os
import pandas as pd
#init
from drugex.logs import logger
from drugex.training.monitors import FileMonitor
#from drugex.logs.utils import initLogger 
from drugex.data.corpus.vocabulary import VocGraph
from drugex.training.generators import GraphTransformer
#processs
from drugex.data.processing import Standardization
#encode
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.data.fragments import FragmentCorpusEncoder
from drugex.data.fragments import GraphFragmentEncoder, FragmentPairsSplitter
#training
from drugex.data.datasets import GraphFragDataSet
#scorer
from drugex.training.scorers.interfaces import Scorer
from rdkit import Chem
from divopt.scoring import BenchmarkScoringFunction
#env
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
#explorer
from drugex.training.explorers import FragGraphExplorer
#custom file monitor
from TimedFileMonitor import TimedFileMonitor
#DElogger to do
#initLogger('datasets.logs')

DATASETS_PATH = 'diverse-hits/optimizers/drugex'

#print(os.listdir(DATASETS_PATH))

df = pd.read_csv(f'{DATASETS_PATH}/jnksmiles.txt', header=None)  #test fragbase is drd2smiles
print(df.head())

#init drugex
GPUS = [0]
MODEL_PATH = f'{DATASETS_PATH}/Papyrus05.5_graph_trans_PT.pkg' 
VOCAB_PATH = f'{DATASETS_PATH}/Papyrus05.5_graph_trans_PT.vocab'
vocab = VocGraph.fromFile(VOCAB_PATH)
pretrained = GraphTransformer(voc_trg=vocab, use_gpus=GPUS)
pretrained.loadStatesFromFile(MODEL_PATH)

#no way this works
smiles_series = df.iloc[:, 0]

#standardization is required for creating the frags
N_PROC = 4 
standardizer = Standardization(n_proc=N_PROC)
smiles = standardizer.apply(smiles_series)
smiles = set(smiles)

#check for processed mols
print(len(smiles))

#encoder
encoder = FragmentCorpusEncoder(
    fragmenter=Fragmenter(4, 4, 'brics'), 
    encoder=GraphFragmentEncoder(
        VocGraph(n_frags=4)
    ),
    pairs_splitter=FragmentPairsSplitter(0.1, len(smiles)), 
    n_proc=N_PROC 
)

#create train and test for training
DATASETS_ENCODED_PATH = f"{DATASETS_PATH}/encoded/graph"
if not os.path.exists(DATASETS_ENCODED_PATH):
    os.makedirs(DATASETS_ENCODED_PATH)

train = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/jnk_train.tsv", rewrite=True)
test = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/jnk_test.tsv", rewrite=True)

#encode and test print
encoder.apply(list(smiles), encodingCollectors=[test, train])
test_from_file = GraphFragDataSet(f'{DATASETS_ENCODED_PATH}/jnk_test.tsv')
print(pd.DataFrame(test_from_file.getData()))

#placeholder for finetune

#Scorer
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

#create ScoringFunctionCall
#constraints
time_budget = 600 #1000
sample_budget = 100000 #10k for sample budget
treshold = [0.2] #0.05<0.3

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

#create env
from drugex.training.rewards import WeightedSum
model_scorer_drd= ModelScorer(scoringFunctionCall("diverse-hits/data/scoring_functions/jnk3"))
env_drd = DrugExEnvironment(scorers=[model_scorer_drd],thresholds=treshold, reward_scheme=WeightedSum())

#Training loader
import copy
from torch.optim import Adam
from drugex.utils import ScheduledOptim
MODEL_FILE_PR = MODEL_PATH
VOCAB_FILE_PR = VOCAB_PATH
vocabulary = VocGraph.fromFile(VOCAB_FILE_PR)
pretrained = GraphTransformer(voc_trg=vocabulary, use_gpus=GPUS)
pretrained.loadStatesFromFile(MODEL_FILE_PR)
mutate  = copy.deepcopy(pretrained).eval().requires_grad_(False) #in case exploration needs a frozen network to explore

#Explorer
#changes:
#- mutate=mutate pretrained performs better
#- treshold = 0.05->0.3->0.5
LR = 1.0
scheduler = ScheduledOptim(Adam(pretrained.parameters(), betas=(0.9,0.98)),init_lr=LR,d_model=512,n_warmup_steps=500)
explorer = FragGraphExplorer(agent=pretrained, env=env_drd, mutate=pretrained, n_samples=2000, epsilon=0.2, use_gpus=GPUS)#, optim= scheduler) #invalid multinomial distribution with plain Adam

#Filemonitor
MODELS_RL_PATH = f'{DATASETS_PATH}/monitor'
#original
"""
monitor = FileMonitor(
    f'{MODELS_RL_PATH}/jnk3_FM', 
    save_smiles=True,
    reset_directory=True, # reset the output directory by removing all previous files
)

"""
#custom FileMonitor
monitor = TimedFileMonitor(
    f"{MODELS_RL_PATH}/jnk3_FM_timeonly",
    time_budget=600,  #600 if time budget        
    save_smiles=True,
    reset_directory=True
)
#test train with custom monitor
N_EPOCHS=10 #5 for sample budget
try:
    explorer.fit(
        train.asDataLoader(batch_size=128),
        test.asDataLoader(batch_size=128),
        monitor=monitor,
        epochs=N_EPOCHS
    )
except TimeoutError as e:
    logger.info(str(e))

if monitor.timed_out():
    logger.info("RL stopped by 600-s timer – proceeding to evaluation.")
#original train
"""
N_EPOCHS = 5
explorer.fit(train.asDataLoader(batch_size=128), test.asDataLoader(batch_size=128), monitor=monitor, epochs=N_EPOCHS)
"""