import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Dataset contains the chain-domain bounds which are indexed by their pdb number
dssp_ss: secondary structure annotation? {H, S, E, T, G, B}
chain-domain-annots: D/N domain/non-domain
"""
df = pd.read_csv('data/ds_final.csv')

plt.title('Distribution of domain length')
plt.hist(df['chain-domain-annots'].apply(lambda x: len(x)), bins=60)
plt.show()

plt.title('count proportion of non-domain residues in each chain')
plt.hist(df['chain-domain-annots'].apply(lambda x: x.count('N') / len(x)), bins=60)
plt.show()

max_residues = 500
n_proteins = len(df)
n_features = 21
feature_df = np.zeros()

bp=True

