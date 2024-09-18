
import pandas as pd
import ast

import sqlite3
import tarfile
import os

# "compound_structures" is a table with 2 409 270 compounds with their molregno and their associated smiles formula
compounds = pd.read_csv('../data/chembl/compound_structures.csv', usecols=['molregno', 'canonical_smiles'])
compounds['canonical_smiles'] = compounds['canonical_smiles'].str.upper()
compounds

# "molecule_synonyms" is a table with the molregno and the synonym associated to the molregno
synonyms = pd.read_csv('../data/chembl/molecule_synonyms.csv', usecols=['molregno', 'synonyms'])
synonyms['synonyms'] = synonyms['synonyms'].str.upper()
synonyms

# Remove duplicate synonyms for each molregno
synonyms_grouped = synonyms.drop_duplicates(subset=['molregno', 'synonyms']).groupby('molregno')['synonyms'].apply(list).reset_index()
synonyms_grouped

# Merge our smiles formulas with synonyms using molregno.
smiles_synonyms = pd.merge(compounds, synonyms_grouped, on='molregno')
smiles_synonyms

# export the dataframe to "chembl_dico.txt" 
smiles_synonyms.to_csv('../data/chembl_dico.txt', sep='\t', index=False, header=False)
