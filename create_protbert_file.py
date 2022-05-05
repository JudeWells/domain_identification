import pandas as pd
df = pd.read_csv('data/ds_final.csv')
df = df[['chain-seq', 'chain-domain-annots']]
df = df.rename(columns = {'chain-seq': 'seq', 'chain-domain-annots':'label'})
df['label'] = df.label.str.replace('N', '0').str.replace('D', '1')
df.to_csv('data/protbert_domain.train.csv', index=False)