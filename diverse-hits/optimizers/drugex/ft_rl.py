import pandas as pd
import sys

sys.path.insert(0, "../drugex")
from drugex.data.processing import Standardization
from drugex.data.fragments import FragmentCorpusEncoder
from drugex.data.fragments import GraphFragmentEncoder, FragmentPairsSplitter
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
import os


def main():
    dfdrd = pd.read_csv("dataset_preparation/drdsmiles.txt", header=None, names=["SMILES"])
    dfgsk = pd.read_csv("dataset_preparation/gsksmiles.txt", header=None, names=["SMILES"]) 
    dfjnk = pd.read_csv("dataset_preparation/jnksmiles.txt", header=None, names=["SMILES"])     

    N_PROC = 4 
    standardizer = Standardization(n_proc=N_PROC)
    dfdrd["SMILES"] = standardizer.apply(dfdrd["SMILES"])
    dfgsk["SMILES"] = standardizer.apply(dfgsk["SMILES"])
    dfjnk["SMILES"] = standardizer.apply(dfjnk["SMILES"])

    #smiles = set(df)
    print(len(dfdrd))
    print(len(dfgsk))
    print(len(dfjnk))

    encoder = FragmentCorpusEncoder(
    fragmenter=Fragmenter(4, 4, 'brics'), 
    encoder=GraphFragmentEncoder(
        VocGraph(n_frags=4) 
    ),
    pairs_splitter=FragmentPairsSplitter(0.1), 
    n_proc=N_PROC 
    )

    #Graphdataset
    DATASETS_PATH = "dataset_preparation"
    DATASETS_ENCODED_PATH = f"{DATASETS_PATH}/encoded/graph"
    if not os.path.exists(DATASETS_ENCODED_PATH):
        os.makedirs(DATASETS_ENCODED_PATH)
    
    traindrd = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/drd_train.tsv", rewrite=True)
    testdrd = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/drd_test.tsv", rewrite=True)
    traingsk = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/gsk_train.tsv", rewrite=True)
    testgsk = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/gsk_test.tsv", rewrite=True)
    trainjnk = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/jnk_train.tsv", rewrite=True)
    testjnk = GraphFragDataSet(f"{DATASETS_ENCODED_PATH}/jnk_test.tsv", rewrite=True)

    encoder.apply(list(dfdrd), encodingCollectors=[testdrd, traindrd])
    encoder.apply(list(dfgsk), encodingCollectors=[testgsk, traingsk])
    encoder.apply(list(dfjnk), encodingCollectors=[testjnk, trainjnk])

    train.getVocPath()
    test.getVocPath()