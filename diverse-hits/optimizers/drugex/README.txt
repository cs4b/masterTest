The DrugEx (version 3.2.0-dev1) model contained in the file named "Papyrus05.5_graph_trans_PT.pkg" was
pretrained with Papyrus version 05.5 (DOI: 10.5281/zenodo.7019874).

The accompanying vocabulary used for the training of this model is contained in the file called
"Papyrus05.5_graph_trans_PT.vocab".

The molecules were parsed from the Papyrus file named "05.5_combined_2D_set_without_stereochemistry.sd.xz"
and converted to SMILES using the RDKit version 2022.03.4 (DOI: 10.5281/zenodo.6798971) with the following code.


Python:
"""

import lzma

from rdkit.Chem import ForwardSDMolSupplier


FILE = "05.5_combined_2D_set_without_stereochemistry.sd.xz"
OUTPUT = "Papyrus05.5_ALL.smi"

# Open file handles
with lzma.open(FILE) as fh, ForwardSDMolSupplier(fh) as supplier, open(OUTPUT, 'w') as oh:
    oh.write('Smiles\n')
    # Iterate over molecules in the file handle
    for mol in supplier:
        if mol is not None:
            oh.write(Chem.MolToSmiles(mol) + '\n')

"""


The resulting data was preprocessed and athe model trained using the commands included under the folders named
"dataset_preparation" and "model_pretraining" respectively.