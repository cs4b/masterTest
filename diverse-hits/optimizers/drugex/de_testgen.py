from drugex.training.generators import GraphTransformer
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet

MODEL_PATH = "diverse-hits/optimizers/drugex/Papyrus05.5_graph_trans_PT.pkg"  # Replace with actual model path
VOCAB_PATH = "diverse-hits/optimizers/drugex/papvoc.vocab"  # Replace with actual vocabulary path

vocab = VocGraph.fromFile(VOCAB_PATH)
GPUS = [0]

pretrained = GraphTransformer(voc_trg=vocab, use_gpus=GPUS)
pretrained.loadStatesFromFile(MODEL_PATH)

num_samples=10
test_from_file = GraphFragDataSet("diverse-hits/optimizers/drugex/A2AR_test.tsv")

generated = pretrained.generate(
    input_dataset=test_from_file,
    num_samples=100, # moleclues to generate
    batch_size=64 # batch size for sampling (choose the highest possible value for your GPU/GPUs)
)
"""
for i,mol in enumerate(generated,1):
    print(f"Molecule{i}: {mol}")"
"""
print(generated.columns)
print(generated.head())

output_file = "generated_molecules.smi"

with open(output_file, "w") as f:
    for i, mol in enumerate(generated["SMILES"], 1):  # Replace "SMILES" with the correct column name if needed
        f.write(f"Molecule{i}: {mol}\n")

print(f"Generated molecules have been written to {output_file}")