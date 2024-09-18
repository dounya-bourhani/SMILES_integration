import pandas as pd
import ast
import os

# ----------- IMPORTATION ----------- #

# Load our chembl and drugbank dico in dataframes
chembl = open('../data/chambl_dico.txt','rt') #open the chembl dico
drugbank = open('../data/drugbank_dico.txt','rt') #open the drugbank dico
df_chembl = pd.read_csv(chembl, sep='\t', names=['id', 'smiles', 'synonyms'])
df_drugbank = pd.read_csv(drugbank, sep='\t', names=['id', 'smiles','synonyms'])
chembl.close()
drugbank.close()

# ----------- PROCESSING ----------- #

# Change the 'synonyms' format to list
df_chembl['synonyms'] = df_chembl['synonyms'].apply(ast.literal_eval)
df_drugbank['synonyms'] = df_drugbank['synonyms'].apply(ast.literal_eval)

# Concatenate both dataframes
dico = pd.concat([df_chembl, df_drugbank], ignore_index=True, join='outer', sort=False)
# Groups the dataframe by the 'smiles' value and aggreagates the 'synonyms' into a list, creating a list of all synonyms corresponding to each unique 'smiles' value
dico = dico.groupby('smiles')['synonyms'].agg(list).reset_index()

# Change format of colum synonyms (from [['ACETAMINOPHEN']['PARACETAMOL']['DOLIPRANE']] to ['ACETAMINOPHEN', 'PARACETAMOL', 'DOLIPRANE'])
for index, row in dico.iterrows(): 
    if type(row['synonyms'][0]) == list :
        row['synonyms'] = [syn for syns in row['synonyms'] for syn in syns] # if already a list we just get all the synonyms and put it in one list
    else :
        row['synonyms'] = [syn for synyonyms_sublist in [ast.literal_eval(syns) for syns in row['synonyms']] for syn in synyonyms_sublist] # if not we change the type of the synonyms to list

# ----------- EXPORTATION ----------- #

# Doesn't work because it truncates very long lists
# dico.to_csv('../data/dico.txt', sep='\t', index=False, header=False)

# Open file in write mode
with open('../data/dico.txt', 'w') as f:
    # Iterate over each row of the DataFrame
    for index, row in dico.iterrows():
        # Write the index, 'smiles' and 'synonyms' columns, separated by tabs
        f.write('\t'.join([str(index), row['smiles'], str(row['synonyms'])]) + '\n')

## Have to remove some unicodes in dico.txt in synonyms list (\u200a, \u200b, \u2028). Did it by hand for now because there are a very few occurences
