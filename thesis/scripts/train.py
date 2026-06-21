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
from ModelScorer import ModelScorer
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

df = pd.read_csv(f'{DATASETS_PATH}/drd2_fragbase.txt', header=None)  #test fragbase is drd2smiles
#print(df.head())

#init drugex
GPUS = [2]
MODEL_PATH = f'{DATASETS_PATH}/Papyrus05.5_graph_trans_PT.pkg' 
VOCAB_PATH = f'{DATASETS_PATH}/Papyrus05.5_graph_trans_PT.vocab'
vocab = VocGraph.fromFile(VOCAB_PATH)
pretrained = GraphTransformer(voc_trg=vocab, use_gpus=GPUS)
pretrained.loadStatesFromFile(MODEL_PATH)

smiles_series = df.iloc[:, 0]

N_PROC = 4 
standardizer = Standardization(n_proc=N_PROC)
smiles = standardizer.apply(smiles_series)
smiles = set(smiles)

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

train = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/drd2_train.tsv", rewrite=True)
test = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/drd2_test.tsv", rewrite=True)

#encode and test print
encoder.apply(list(smiles), encodingCollectors=[test, train])
test_from_file = GraphFragDataSet(f'{DATASETS_ENCODED_PATH}/drd2_test.tsv')
print(pd.DataFrame(test_from_file.getData()))

#constraints
time_budget = 10000 #10000 for sample constrain / 600 for time
sample_budget = 100
treshold = [0.7] #0.05<0.3

#scoringFunctionCall
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
from drugex.training.rewards import SingleReward #setup single objective scoring function scheme
model_scorer_drd= ModelScorer(scoringFunctionCall("diverse-hits/data/scoring_functions/drd2"), time_budget=time_budget,sample_budget=sample_budget)
model_scorer_drd.reset_budgets()
env_drd = DrugExEnvironment(scorers=[model_scorer_drd],thresholds=treshold, reward_scheme=SingleReward())

#Training loader
#import copy
from torch.optim import Adam

MODEL_FILE_PR = MODEL_PATH
VOCAB_FILE_PR = VOCAB_PATH
vocabulary = VocGraph.fromFile(VOCAB_FILE_PR)
#pretrained = GraphTransformer(voc_trg=vocabulary, use_gpus=GPUS) #already called
pretrained.loadStatesFromFile(MODEL_FILE_PR)
#mutate  = copy.deepcopy(pretrained).eval().requires_grad_(False) #drugex does this implicitely so not needed

#explorer
explorer = FragGraphExplorer(agent=pretrained, env=env_drd, mutate=pretrained, n_samples=2000, epsilon=0.2, use_gpus=GPUS)

#custom monitor
MODELS_RL_PATH = f'{DATASETS_PATH}/monitor'
monitor2 = TimedFileMonitor(
    f"{MODELS_RL_PATH}/drd2_FM_onlytime",
    time_budget=100,          
    save_smiles=True,
    reset_directory=True
)
#default monitor
monitor = FileMonitor(
    f'{DATASETS_PATH}/drd2_time_test', 
    save_smiles=True,
    reset_directory=True, 
)

#train
N_EPOCHS=10
try:
    explorer.fit(
        train.asDataLoader(batch_size=128),
        test.asDataLoader(batch_size=128),
        monitor=monitor,
        epochs=N_EPOCHS
    )
except TimeoutError as e:
    logger.info(str(e))
    print(f'{e}')
except RuntimeError as r:
    logger.info(str(r))
    print(f'{r}')

#if monitor.timed_out():
#    logger.info("RL stopped by 600-s timer – proceeding to evaluation.")
#    print("stopped by timelimit")
