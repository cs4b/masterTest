import pandas as pd

df = pd.read_csv("diverse-hits/optimizers/drugex/splits.csv")  
df_label1 = df[df["label"] == 1]
df_label1["smiles"].to_csv("diverse-hits/optimizers/drugex/fragbase.txt", index=False, header=False)