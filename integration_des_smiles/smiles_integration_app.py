import re
import os
import ast
import sys
import time
import json
import torch  
torch.cuda.empty_cache()
import spacy
import edsnlp
import string
import requests
import polars as pl
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from unidecode import unidecode
from datasets import Dataset, load_from_disk

# import warnings
# # Ignore the warnings
# warnings.simplefilter("ignore")

# os.environ['CURL_CA_BUNDLE'] = ''
# import urllib3
# # Deactivate the SSL warnings
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def is_only_digit_and_punctuation (word): #returns true if the given string contains only digits, punctuation or space
    boolean = True
    for i in range(len(word)):
        if word[i] not in string.digits and word[i] not in string.punctuation and word[i]!=' ':
            boolean = False
    return (boolean) 

def find_drugs_edsnlp(doc): # returns a list of all the drugs recognized by the eds-nlp library
    drugs_to_replace=[]
    
    for X in doc.ents : #for each tagged string
        if len((X.text).split(' ')) < 4 and is_only_digit_and_punctuation(X.text)==False:
        # if the string is lower than 4 words and not only digits or punctuation
        # (to avoid tagged strings that are not drugs in reality, but just phone number or random strings)
            drugs_to_replace.append(unidecode(X.text.upper())) #add the tagged string to the list, putting it in upper case and removing accents

    return drugs_to_replace

def get_drugs_with_edsnlp(reports):
    # new_patient_reports = pd.DataFrame(pd.read_excel(file))
    # reports = pd.read_excel(file, engine='openpyxl')
    
    # Add spacy pipelines to detect our drugs in our 
    nlp = edsnlp.blank("eds")

    nlp.add_pipe("eds.normalizer")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe("eds.drugs")
    
    detected_drugs = []
    for text in tqdm(reports['text']):
        if type(text)==str : #in case of an empty row 
            doc = nlp(text)
            drugs_to_replace = find_drugs_edsnlp(doc) #creates a list with all the words taggued as a drug
            if len(drugs_to_replace) > 0 : 
                detected_drugs.append(sorted(drugs_to_replace)) #if drugs have been detected in this report we add the list 
            else:
                detected_drugs.append([]) #else we add an empty list

    reports['drugs'] = detected_drugs  
    return reports

#takes the name of a drug as input and search its SMILES formula in the chembl + drugbank dictionnary
#returns the SMILES formula if it exists and "None" if it doesn't
def find_SMILES_formula(df_dico, drug_name):
    SMILES_formula = None   

    # Define the strategies to go search in the drugs lists
    search_strategies = [
            drug_name,
            ' '.join(drug_name.split('-')),
            '-'.join(drug_name.split(' ')),
            drug_name.rstrip('E') if drug_name[-1] == 'E' else None,
            drug_name[:-1] + 'A' if drug_name[-1] == 'E' else None
        ]

    # get the drugs which contains the drug_name in their drug lists. We look for every strategy.
    for strategy in search_strategies:
        if strategy is not None:
            result = df_dico.filter(pl.col('drugs').list.contains(strategy))
            if not result.is_empty():
                SMILES_formula = result.select('smiles').to_series()[0]
                break

    return SMILES_formula

def get_smiles(reports):
    df_dico = pl.read_csv('../data/dico.txt', separator='\t', new_columns=['id', 'smiles', 'drugs'])    #open the dico in a polars Dataframe
    df_dico = df_dico.with_columns(drugs = pl.col('drugs').map_elements(eval, return_dtype=pl.List(pl.Utf8)))     # put our drugs in lists (initialy strings)

    smiles = [] # get smiles formula from all our detected drugs
    for report_drugs in tqdm(reports['drugs']):
        report_smiles = []  # smiles for the report we are in
        for drug in report_drugs:
            formula = find_SMILES_formula(df_dico, drug)
            report_smiles.append(formula)
        smiles.append(report_smiles)
    reports['smiles'] = smiles  # add the smiles to our dataframe
    return reports

# Replace the drugs names by thier corresponding SMILES formula. Here the reports source is a dataset 
def replace_drugs_by_smiles_dataset(row):
    rich_text = row['text']                 # report text
    drugs = row['drugs']      # list of the drugs in the reports
    smiles = row['smiles']    # list of the smiles for each drug

    rich_text = unidecode(rich_text.lower())
    # replace each drug name by its smiles formula in the text
    for i, drug in enumerate(drugs):    
        if smiles[i] is not None:
            rich_text = rich_text.replace(drug.lower(), smiles[i])

    # return rich_text
    return {"enriched_text": rich_text}

def create_enriched_reports_dataset(reports, save_directory):
    # Apply the function to each row of the DataFrame
    enriched_dataset = reports.map(replace_drugs_by_smiles_dataset, desc="Replacing drug name by their smiles")  
    # save the result
    enriched_dataset.save_to_disk(save_directory)

    return enriched_dataset


def main():
    os.chdir("/home/bourhani@clb.loc/SMILES/scripts")

    start_time = time.time()

    # dataset that I want to integrate the SMILES formula in 
    dataset = load_from_disk("/home/bourhani@clb.loc/saepred/data_test/featurized/OncoBERT_nobias_2LAB/test")

    print("----------------------------- DRUGS DETECTION -----------------------------")
    # detect the drugs in the reports and return a dataframe with the drugs in each text
    drugs_dataset = dataset.map(get_drugs_with_edsnlp, batched=True, desc="Detect the drugs in the reports")

    print("----------------------------- SMILES TRANSFORMATION -----------------------------")
    # find their corresponding smiles and return a dataframe with smiles for each drugs
    smiles_dataset = drugs_dataset.map(get_smiles, batched=True, desc="Get the smiles formula for each drugs")

    print("----------------------------- SMILES INTEGRATION -----------------------------")
    # create the reports enriched with the smiles
    save_path = "/home/bourhani@clb.loc/saepred/data_test/featurized/OncoBERT_nobias_2LAB_smiles/test"
    enriched_dataset = create_enriched_reports_dataset(smiles_dataset, save_path)
    print(enriched_dataset)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Convert elapsed time to hours, minutes and seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTemps d'exécution : {int(hours)} heures, {int(minutes)} minutes et {seconds:.2f} secondes")

if __name__ == "__main__":
    main()