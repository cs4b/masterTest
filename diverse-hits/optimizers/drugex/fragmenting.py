import pandas as pd
from drugex.data.processing import Standardization

#read
df = pd.read_csv("diverse-hits/optimizers/drugex/fragbase.txt")  
print(df.head())

#standardize
N_PROC = 4 # number of cpus
standardizer = Standardization(n_proc=N_PROC)
smiles = standardizer.apply(df)

len(df)

#scipy 1.11.1 pypi_0 pypi updated